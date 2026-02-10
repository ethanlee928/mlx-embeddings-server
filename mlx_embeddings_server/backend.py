import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import mlx.core as mx
import requests
from mlx_embeddings.utils import load
from PIL import Image

logger = logging.getLogger("uvicorn.error")


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


class BaseModel(ABC):
    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor

    @abstractmethod
    def get_embeddings(self, inputs: List[str]) -> List[Any]:
        pass

    def _prepare_inputs(self, inputs: List[str]) -> Tuple[List[int], List[str], List[int], List[Image.Image]]:
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
                    logger.info(f"Input {idx} detected as an image.")
                except Exception as e:
                    logger.warning(f"Failed to load image input {idx}: {e}")

            if not is_img:
                text_indices.append(idx)
                text_inputs_list.append(inp)

        return text_indices, text_inputs_list, image_indices, image_inputs_list


class ColQwenModel(BaseModel):
    def get_embeddings(self, inputs: List[str]) -> List[Any]:
        results: List[Any] = [None] * len(inputs)
        text_indices, text_inputs, image_indices, image_inputs = self._prepare_inputs(inputs)

        # Process text
        if text_inputs:
            text_inputs_proc = self.processor(text=text_inputs, padding=True, return_tensors="pt")
            text_input_ids = mx.array(text_inputs_proc.input_ids)

            proc_out = self.model(input_ids=text_input_ids)
            embeds_list = proc_out.text_embeds.tolist()

            for i, embed in zip(text_indices, embeds_list):
                results[i] = embed

        # Process images
        if image_inputs:
            text_prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image."
            prompts = [text_prompt] * len(image_inputs)

            image_inputs_proc = self.processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt")

            image_input_ids = mx.array(image_inputs_proc.input_ids)
            pixel_values = mx.array(image_inputs_proc.pixel_values)
            image_grid_thw = mx.array(image_inputs_proc.image_grid_thw)

            proc_out = self.model(input_ids=image_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

            embeds_list = proc_out.image_embeds.tolist()

            for i, embed in zip(image_indices, embeds_list):
                results[i] = embed

        return results


class SigLIPModel(BaseModel):
    def get_embeddings(self, inputs: List[str]) -> List[Any]:
        results: List[Any] = [None] * len(inputs)
        text_indices, text_inputs, image_indices, image_inputs = self._prepare_inputs(inputs)
        # Process text
        if text_inputs:
            text_inputs_proc = self.processor(text=text_inputs, padding="max_length", return_tensors="pt")
            text_input_ids = mx.array(text_inputs_proc.input_ids)

            # Standard MLX model usage for embeddings
            proc_out = self.model.get_text_features(input_ids=text_input_ids)
            embeds_list = proc_out.tolist()

            for i, embed in zip(text_indices, embeds_list):
                results[i] = embed

        # Process images
        if image_inputs:
            image_inputs_proc = self.processor(images=image_inputs, padding="max_length", return_tensors="pt")
            # SigLIP expects NHWC format (Batch, Height, Width, Channels)
            pixel_values = mx.array(image_inputs_proc.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)

            proc_out = self.model.get_image_features(pixel_values=pixel_values)
            embeds_list = proc_out.tolist()

            for i, embed in zip(image_indices, embeds_list):
                results[i] = embed

        return results


class ModelManager:
    _instance = None

    def __init__(self):
        self.model_id = os.getenv("MODEL_ID")
        if not self.model_id:
            raise ValueError("MODEL_ID environment variable is not set. Please specify the model to load.")

        logger.info(f"Loading model: {self.model_id}")
        model, processor = load(self.model_id)

        # Determine model engine
        model_type = getattr(model.config, "model_type", None)
        logger.info(f"Model type detected: {model_type}")

        if model_type == "siglip":
            self.engine = SigLIPModel(model, processor)
        else:
            # Defaulting to ColQwen for other types for now, or we can be more specific
            self.engine = ColQwenModel(model, processor)

        logger.info(f"Loading model: {self.model_id} [COMPLETED]")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_embeddings(inputs: List[str]) -> List[Any]:
    manager = ModelManager.get_instance()
    return manager.engine.get_embeddings(inputs)
