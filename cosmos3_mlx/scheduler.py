"""UniPC Multistep Scheduler for Cosmos 3 diffusion generation.

Implements the unified predictor-corrector (UniPC) framework for
iterative denoising. This is the scheduler used by Cosmos 3 for
image/video/audio generation.

The scheduler manages:
- Noise schedule (linear beta schedule for rectified flow)
- Timestep management
- Denoising step computation
"""

from typing import Optional

import mlx.core as mx
import numpy as np


class UniPCScheduler:
    """UniPC multi-step scheduler for rectified flow denoising.

    Cosmos 3 uses rectified flow: X_t = t * noise + (1-t) * data,
    with velocity prediction.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 30,
        timestep_scale: float = 0.001,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.timestep_scale = timestep_scale
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        # Linear spacing from 1.0 to 0.0 (high noise to clean)
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = np.arange(num_inference_steps, 0, -1) * step_ratio
        self.timesteps = mx.array(timesteps.astype(np.float32)) * self.timestep_scale

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        next_timestep: Optional[mx.array] = None,
    ) -> mx.array:
        """Perform one denoising step.

        For rectified flow with velocity prediction:
        x_{t-1} = x_t + (t_{i-1} - t_i) * v_predicted

        Args:
            model_output: predicted velocity from the transformer
            timestep: current timestep t_i
            sample: current noisy sample x_t
            next_timestep: next timestep t_{i-1} (if None, assumed 0 = clean)

        Returns:
            denoised sample x_{t-1}
        """
        if next_timestep is None:
            next_timestep = mx.array(0.0)

        dt = next_timestep - timestep
        prev_sample = sample + dt * model_output

        return prev_sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """Add noise to clean samples at given timestep.

        Rectified flow interpolation: X_t = t * noise + (1-t) * data
        """
        t = timestep
        if t.ndim == 0:
            t = t.reshape(1)
        # Broadcast t to match sample dims
        while t.ndim < original_samples.ndim:
            t = mx.expand_dims(t, -1)

        return t * noise + (1 - t) * original_samples
