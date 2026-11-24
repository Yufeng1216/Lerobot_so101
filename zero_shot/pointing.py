"""Ask a local vision-language model (Qwen3-VL-2B-Instruct) where things are in the overhead image.

The model is prompted to answer with a point as JSON, e.g. [{"point_2d": [512, 233], "label": "toy"}].
Qwen3-VL gives coordinates on a 0-1000 scale relative to the image (not pixels), so they are
rescaled to the 640x480 frame. Runs fully on the laptop GPU (~4.4 GB weights in bf16): the room
images never leave the machine.

Usage (inside the `lerobot` conda env):
    python -m zero_shot.pointing path/to/image.jpg "the blue plush toy"
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = ('Point to {target} in the image. Answer only with JSON in this format: '
          '[{{"point_2d": [x, y], "label": "{label}"}}]')


class Pointer:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device).eval()

    @torch.no_grad()
    def point(self, image_rgb: np.ndarray, target: str, label: str = "target") -> tuple[tuple[float, float] | None, str, float]:
        """((u, v) in pixels or None, raw model answer, seconds)."""
        h, w = image_rgb.shape[:2]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image_rgb)},
            {"type": "text", "text": PROMPT.format(target=target, label=label)},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        t = time.time()
        out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_point(answer, w, h), answer, time.time() - t


def parse_point(answer: str, width: int, height: int) -> tuple[float, float] | None:
    """Pixel (u, v) from the model's JSON answer (0-1000 scale). Falls back to a bbox centre."""
    m = re.search(r"\[.*\]", answer, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    for it in items if isinstance(items, list) else [items]:
        if isinstance(it, dict) and "point_2d" in it:
            x, y = it["point_2d"][:2]
        elif isinstance(it, dict) and "bbox_2d" in it:
            x1, y1, x2, y2 = it["bbox_2d"][:4]
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            continue
        return x / 1000 * width, y / 1000 * height
    return None


if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    p = Pointer()
    print(p.point(img, sys.argv[2]))
