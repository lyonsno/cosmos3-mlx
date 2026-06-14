"""End-to-end inference for Cosmos 3 Nano on MLX.

Phase 1: Text-only and text+image understanding (AR reasoner).
Usage:
    python -m cosmos3_mlx.generate --model-dir weights/Cosmos3-Nano --prompt "Hello"
    python -m cosmos3_mlx.generate --model-dir weights/Cosmos3-Nano --image photo.jpg --prompt "What is in this image?"
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .load import load_transformer, load_vision_encoder, load_tokenizer
from .model import Cosmos3Model
from .vision import VisionModel


def preprocess_image(
    image_path: str,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
) -> tuple[mx.array, mx.array]:
    """Load and preprocess an image for the vision encoder.

    Args:
        image_path: path to image file
        patch_size: spatial patch size
        temporal_patch_size: temporal patch size

    Returns:
        pixel_values: [1, 3, temporal_patch_size, H, W] normalized
        grid_thw: [1, 3] grid dimensions
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Resize to nearest multiple of patch_size
    w, h = img.size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    if new_h == 0:
        new_h = patch_size
    if new_w == 0:
        new_w = patch_size
    img = img.resize((new_w, new_h))

    # Convert to numpy array and normalize
    pixels = np.array(img).astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pixels = (pixels - mean) / std

    # Reshape: [H, W, 3] -> [1, 3, temporal_patch_size, H, W]
    pixels = pixels.transpose(2, 0, 1)  # [3, H, W]
    # Repeat along temporal dimension to match temporal_patch_size
    pixels = np.stack([pixels] * temporal_patch_size, axis=1)  # [3, T, H, W]
    pixels = pixels[np.newaxis, ...]  # [1, 3, T, H, W]

    pixel_values = mx.array(pixels)

    # Grid dimensions after patching
    t_patches = temporal_patch_size // temporal_patch_size  # = 1
    h_patches = new_h // patch_size
    w_patches = new_w // patch_size
    grid_thw = mx.array([[t_patches, h_patches, w_patches]])

    return pixel_values, grid_thw


def generate_text(
    model: Cosmos3Model,
    tokenizer,
    prompt: str,
    vision_embeds: Optional[mx.array] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate text from a prompt, optionally with vision embeddings.

    Args:
        model: loaded Cosmos3Model
        tokenizer: loaded tokenizer
        prompt: text prompt
        vision_embeds: optional vision encoder output [num_patches, hidden_size]
        max_tokens: maximum tokens to generate
        temperature: sampling temperature (0 = greedy)

    Returns:
        generated text string
    """
    # Build chat-formatted input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text, return_tensors=None)
    input_ids = mx.array([input_ids])

    # Generate
    if vision_embeds is not None:
        # TODO: Insert vision embeddings at the right position
        # For now, just do text-only generation
        pass

    output_ids = model.generate(
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    mx.eval(output_ids)

    # Decode only the generated part
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def main():
    parser = argparse.ArgumentParser(description="Cosmos 3 Nano MLX inference")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--reasoner-only",
        action="store_true",
        default=True,
        help="Load only reasoner weights (default: True)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist.")
        print("Download weights first:")
        print(f"  huggingface-cli download nvidia/Cosmos3-Nano --local-dir {model_dir}")
        return

    print(f"Loading Cosmos 3 Nano from {model_dir}...")

    # Load transformer
    t0 = time.time()
    print("  Loading transformer weights...")
    model = load_transformer(model_dir, reasoner_only=args.reasoner_only)
    t1 = time.time()
    print(f"  Transformer loaded in {t1 - t0:.1f}s")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = load_tokenizer(model_dir)
    t2 = time.time()
    print(f"  Tokenizer loaded in {t2 - t1:.1f}s")

    # Optionally load vision encoder and process image
    vision_embeds = None
    if args.image:
        print("  Loading vision encoder...")
        vision_model = load_vision_encoder(model_dir)
        t3 = time.time()
        print(f"  Vision encoder loaded in {t3 - t2:.1f}s")

        print(f"  Processing image: {args.image}")
        pixel_values, grid_thw = preprocess_image(args.image)
        vision_embeds = vision_model(pixel_values, grid_thw)
        mx.eval(vision_embeds)
        t4 = time.time()
        print(f"  Image processed in {t4 - t3:.1f}s ({vision_embeds.shape[0]} patches)")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("Generating...\n")

    t_gen = time.time()
    response = generate_text(
        model,
        tokenizer,
        args.prompt,
        vision_embeds=vision_embeds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    t_done = time.time()

    print(f"Response: {response}")
    print(f"\n[Generated in {t_done - t_gen:.1f}s]")


if __name__ == "__main__":
    main()
