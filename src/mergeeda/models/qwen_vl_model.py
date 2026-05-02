"""Qwen Vision-Language Model Wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenVLModel:
    """Qwen Vision-Language Model for multimodal question answering."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """Initialize the Qwen3-VL model from a HuggingFace model identifier."""
        if not model_name:
            raise ValueError("model_name must be provided")

        self.modality = "multimodal"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(
        self, question: str, imgs: Optional[list[Image.Image]] = None
    ) -> str:
        """Generate an answer to a question with optional images."""
        with torch.no_grad():
            if imgs is None:
                imgs = []

            content = []
            for img in imgs:
                content.append({"type": "image", "image": img})

            content.append({"type": "text", "text": question})

            system_prompt = (
                "Answer in plain text only. "
                "Do not use markdown formatting such as headers, bullet points, bold, or italics. "
                "LaTeX math expressions and code blocks are allowed."
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": content},
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            inputs.pop("token_type_ids", None)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0]
