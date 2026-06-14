"""UniPC Multistep Scheduler for Cosmos 3 diffusion generation.

Implements the unified predictor-corrector (UniPC) framework for
iterative denoising with flow matching. Matches the HuggingFace
UniPCMultistepScheduler with flow_prediction type.

Key formulas (flow matching path):
- sigma_t is the noise level (0 = clean, 1 = pure noise)
- alpha_t = 1 - sigma_t
- x_t = alpha_t * x_0 + sigma_t * noise
- Model predicts velocity v, x_0 recovered as: x_0 = x_t - sigma_t * v
- Stepping uses log-SNR (lambda) space exponential integrators
"""

from typing import Optional

import mlx.core as mx
import numpy as np


class UniPCScheduler:
    """UniPC multi-step scheduler for flow matching denoising.

    Matches HF UniPCMultistepScheduler config:
    - prediction_type: flow_prediction
    - use_flow_sigmas: True
    - predict_x0: True
    - solver_order: 2
    - solver_type: bh2
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        flow_shift: float = 1.0,
        solver_order: int = 2,
        use_karras_sigmas: bool = True,
        sigma_min: float = 0.147,
        sigma_max: float = 200.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.flow_shift = flow_shift
        self.solver_order = solver_order
        self.use_karras_sigmas = use_karras_sigmas
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigmas = None
        self.timesteps = None
        self.model_outputs = []  # History for multistep
        self.step_index = 0

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference.

        Uses Karras sigma schedule (matching HF config: use_karras_sigmas=True,
        use_flow_sigmas=True) converted to flow-matching space.
        """
        self.num_inference_steps = num_inference_steps

        if self.use_karras_sigmas:
            # Karras sigma schedule: concentrate steps at lower noise for detail
            # rho=7 is the standard Karras value
            rho = 7.0
            ramp = np.linspace(0, 1, num_inference_steps)
            min_inv_rho = self.sigma_min ** (1 / rho)
            max_inv_rho = self.sigma_max ** (1 / rho)
            karras_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

            # Convert Karras sigmas to flow-matching space: σ_flow = σ / (σ + 1)
            sigmas = karras_sigmas / (karras_sigmas + 1)
        else:
            # Flow sigmas: linear from 1 to near-zero
            sigmas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)[:-1]

            # Apply flow shift
            if self.flow_shift != 1.0:
                sigmas = self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)

        # Ensure sigma[0] is not exactly 1.0 (causes log(0) in lambda)
        eps = 1e-6
        if abs(sigmas[0] - 1.0) < eps:
            sigmas[0] -= eps

        # Append terminal sigma = 0
        sigmas = np.append(sigmas, 0.0)

        self.sigmas = mx.array(sigmas.astype(np.float32))
        self.timesteps = mx.array((sigmas[:-1] * self.num_train_timesteps).astype(np.float32))

        # Reset state
        self.model_outputs = []
        self.step_index = 0

    def _convert_to_x0(
        self,
        model_output: mx.array,
        sigma: float,
        sample: mx.array,
    ) -> mx.array:
        """Convert flow prediction to x0 prediction.

        For flow matching: x_0 = x_t - sigma_t * v_predicted
        """
        return sample - sigma * model_output

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        next_timestep: Optional[mx.array] = None,
    ) -> mx.array:
        """Perform one denoising step using UniPC.

        Args:
            model_output: predicted velocity from the transformer
            timestep: current timestep (unused, we use step_index)
            sample: current noisy sample x_t
            next_timestep: unused (we use step_index)

        Returns:
            denoised sample x_{t-1}
        """
        sigma_s = float(self.sigmas[self.step_index].item())
        sigma_t = float(self.sigmas[self.step_index + 1].item())

        # Convert model output to x0 prediction
        x0_pred = self._convert_to_x0(model_output, sigma_s, sample)

        # Store for multistep
        self.model_outputs.append(x0_pred)
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)

        # Compute alpha and sigma for source and target
        alpha_s = 1.0 - sigma_s
        alpha_t = 1.0 - sigma_t

        if sigma_t == 0.0:
            # Final step: just return x0
            result = x0_pred
        elif len(self.model_outputs) == 1 or self.solver_order == 1:
            # First-order step (or order-1 mode)
            result = self._first_order_step(
                x0_pred, sample, sigma_s, sigma_t, alpha_s, alpha_t
            )
        else:
            # Second-order multistep
            result = self._second_order_step(
                self.model_outputs, sample, sigma_s, sigma_t, alpha_s, alpha_t
            )

        self.step_index += 1
        return result

    def _first_order_step(
        self,
        x0_pred: mx.array,
        sample: mx.array,
        sigma_s: float,
        sigma_t: float,
        alpha_s: float,
        alpha_t: float,
    ) -> mx.array:
        """First-order UniPC step in log-SNR space.

        lambda = log(alpha/sigma)
        h = lambda_t - lambda_s
        x_t = (sigma_t/sigma_s) * x_s - alpha_t * (e^(-h) - 1) * x0_pred
        """
        # Log-SNR values
        lambda_s = np.log(alpha_s / max(sigma_s, 1e-10))
        lambda_t = np.log(max(alpha_t, 1e-10) / max(sigma_t, 1e-10))
        h = lambda_t - lambda_s

        # For predict_x0 with bh2 solver:
        # x_t = (sigma_t / sigma_s) * x_s - alpha_t * (exp(-h) - 1) * x0
        h_phi_1 = np.expm1(-h)  # e^(-h) - 1

        x_t = (sigma_t / sigma_s) * sample - alpha_t * h_phi_1 * x0_pred
        return x_t

    def _second_order_step(
        self,
        model_outputs: list,
        sample: mx.array,
        sigma_s: float,
        sigma_t: float,
        alpha_s: float,
        alpha_t: float,
    ) -> mx.array:
        """Second-order UniPC multistep in log-SNR space.

        Uses the previous x0 prediction to compute a correction term.
        """
        m0 = model_outputs[-1]  # Latest x0 prediction
        m_prev = model_outputs[-2]  # Previous x0 prediction

        # Get previous sigma for the D1 coefficient
        prev_step = max(0, self.step_index - 1)
        sigma_prev = float(self.sigmas[prev_step].item())
        alpha_prev = 1.0 - sigma_prev

        # Log-SNR values
        lambda_s = np.log(alpha_s / max(sigma_s, 1e-10))
        lambda_t = np.log(max(alpha_t, 1e-10) / max(sigma_t, 1e-10))
        lambda_prev = np.log(alpha_prev / max(sigma_prev, 1e-10))

        h = lambda_t - lambda_s
        h_prev = lambda_s - lambda_prev
        r = h_prev / h if abs(h) > 1e-10 else 0.0

        # First-order base
        h_phi_1 = np.expm1(-h)
        x_t_ = (sigma_t / sigma_s) * sample - alpha_t * h_phi_1 * m0

        # Second-order correction: D1 = (m0 - m_prev), weighted by r
        # For bh2 solver: B_h = expm1(-h)
        B_h = np.expm1(-h)

        # rho coefficient for second order
        rho = 1.0 / (2.0 * r) if abs(r) > 1e-10 else 0.0

        D1 = m0 - m_prev
        x_t = x_t_ - alpha_t * B_h * rho * D1

        return x_t

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """Add noise to clean samples at given timestep.

        Flow matching: X_t = (1 - sigma) * data + sigma * noise
        """
        sigma = timestep / self.num_train_timesteps
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        while sigma.ndim < original_samples.ndim:
            sigma = mx.expand_dims(sigma, -1)

        return (1 - sigma) * original_samples + sigma * noise
