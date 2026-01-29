import base64
import io
import logging
import os
from typing import Any, List

import mlx.core as mx
import requests
from mlx_embeddings.utils import load
from PIL import Image

logger = logging.getLogger("uvicorn.error")


class ModelManager:
    _instance = None

    def __init__(self):
        self.model_id = os.getenv("MODEL_ID")
        if not self.model_id:
            raise ValueError("MODEL_ID environment variable is not set. Please specify the model to load.")

        logger.info(f"Loading model: {self.model_id}")
        self.model, self.processor = load(self.model_id)
        logger.info(f"Loading model: {self.model_id} [COMPLETED]")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def is_url(string: str) -> bool:
    return string.startswith("http://") or string.startswith("https://")


def is_base64_image(string: str) -> bool:
    return string.startswith("data:image")


def load_image(image_str: str) -> Image.Image:
    if is_url(image_str):
        resp = requests.get(image_str, stream=True)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    elif is_base64_image(image_str):
        try:
            header, encoded = image_str.split(",", 1)
            data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")
    else:
        try:
            data = base64.b64decode(image_str)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not decode image: {e}")


def get_embeddings(inputs: List[str]) -> List[List[List[float]]]:
    manager = ModelManager.get_instance()
    model = manager.model
    processor = manager.processor

    results: List[Any] = [None] * len(inputs)

    text_indices = []
    text_inputs_list = []

    image_indices = []
    image_inputs_list = []

    for idx, inp in enumerate(inputs):
        is_img = False
        if is_url(inp) or is_base64_image(inp):
            try:
                img = load_image(inp)
                image_indices.append(idx)
                image_inputs_list.append(img)
                is_img = True
                logger.info(f"Input {idx} detected as image.")
            except Exception as e:
                logger.warning(f"Failed to load image input {idx}: {e}")

        if not is_img:
            text_indices.append(idx)
            text_inputs_list.append(inp)

    # Process text
    if text_inputs_list:
        text_inputs_proc = processor(text=text_inputs_list, padding=True, return_tensors="pt")
        text_input_ids = mx.array(text_inputs_proc.input_ids)

        proc_out = model(input_ids=text_input_ids)
        embeds_list = proc_out.text_embeds.tolist()

        for i, embed in zip(text_indices, embeds_list):
            results[i] = embed

    # Process images
    if image_inputs_list:
        # Standard strict prompt for ColQwen
        text_prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image."
        prompts = [text_prompt] * len(image_inputs_list)

        image_inputs_proc = processor(text=prompts, images=image_inputs_list, padding=True, return_tensors="pt")

        image_input_ids = mx.array(image_inputs_proc.input_ids)
        pixel_values = mx.array(image_inputs_proc.pixel_values)
        image_grid_thw = mx.array(image_inputs_proc.image_grid_thw)

        proc_out = model(input_ids=image_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        embeds_list = proc_out.image_embeds.tolist()

        for i, embed in zip(image_indices, embeds_list):
            results[i] = embed

    return results
