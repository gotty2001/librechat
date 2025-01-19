"""
title: Huggingface Pipe
author: Wes Caldwell
email: Musicheardworldwide@gmail.com
date: 2024-07-19
version: 1.05
license: MIT
description: Function to use HF models on Open-WebUI
requirements: pydantic, huggingface-hub, transformers
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator, List
import os
import logging
from huggingface_hub import HfApi
from transformers import pipeline, set_seed, PipelineException

logging.basicConfig(level=logging.INFO)


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="HUGGINGFACE/",
            description="Prefix to be added before model names.",
        )
        HUGGINGFACE_API_URL: str = Field(
            default="https://api-inference.huggingface.co/models/",
            description="Base URL for accessing Hugging Face API endpoints.",
        )
        HUGGINGFACE_API_KEY: str = Field(
            default=os.getenv("HUGGINGFACE_API_KEY", ""),
            description="API key for authenticating requests to the Hugging Face API.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.hf_api = HfApi()

    def fetch_models(self) -> List[dict]:
        """Fetch models from Hugging Face containing 'gpt' in their ID."""
        if not self.valves.HUGGINGFACE_API_KEY:
            logging.error("API Key not provided.")
            return [{"id": "error", "name": "API Key not provided."}]

        try:
            models = self.hf_api.list_models(
                use_auth_token=self.valves.HUGGINGFACE_API_KEY
            )
            filtered_models = [
                {
                    "id": model.modelId,
                    "name": f"{self.valves.NAME_PREFIX}{model.modelId}",
                }
                for model in models
                if "gpt" in model.modelId
            ]
            return filtered_models

        except Exception as e:
            logging.error(f"Failed to fetch models: {e}")
            return [
                {"id": "error", "name": "Could not fetch models. Update the API Key."}
            ]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Generator, Iterator]:
        """Process a text-generation request."""
        if "model" not in body:
            logging.error("Model not specified in the request body.")
            return "Error: Model not specified in the request body."

        if "prompt" not in body:
            logging.error("Prompt not provided in the request body.")
            return "Error: Prompt not provided in the request body."

        model_id = body["model"]
        logging.info(f"Processing request for model: {model_id}")

        try:
            # Set up the text generation pipeline
            device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
            generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=device,
            )

            set_seed(42)  # Optional: Ensure reproducibility

            response = generator(
                body["prompt"],
                max_length=body.get("max_tokens", 50),
                num_return_sequences=1,
                do_sample=True,
            )

            if body.get("stream", False):
                # Stream results as a generator
                for message in response:
                    yield message["generated_text"]
            else:
                # Return the entire response
                return [msg["generated_text"] for msg in response]

        except PipelineException as pe:
            logging.error(f"Pipeline error: {pe}")
            return f"Error: {pe}"
        except Exception as e:
            logging.error(f"Failed to process request: {e}")
            return f"Error: {e}"


if __name__ == "__main__":
    pipe = Pipe()

    # Test fetch_models
    models = pipe.fetch_models()
    print(models)

    # Test pipe method
    test_body = {
        "model": "gpt2",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "stream": False,
    }
    result = pipe.pipe(test_body, __user__={})

    if isinstance(result, str):
        print(result)
    else:
        for res in result:
            print(res)
