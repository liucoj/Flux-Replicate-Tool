"""
title: Flux DEV on Replicate.com 
description: Generate images with Flux DEV hosted on Replicate.com
author: Liu_C0j
version: 1.0
"""

from typing import (
    Literal,
)
import requests
import os
import json
from pydantic import BaseModel, Field

# image correctly appears and description is well formatted
ImageAspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]


class Tools:
    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="",
            description="Your Replicate API token",
        )

    def __init__(self):
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
        )

    async def generate_image(
        self,
        image_prompt: str,
        image_aspect_ratio: ImageAspectRatioType,
        __event_emitter__=None,
    ) -> str:
        """
        :param image_prompt: Text prompt for image generation.
        :param image_aspect_ratio: Aspect ratio for the generated image.
        """

        try:
            # Emitting event twice is needed by Replicate API
            for _ in range(2):
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generating image ...", "done": False},
                    }
                )

            replicate_api_token = self.valves.REPLICATE_API_TOKEN
            if not replicate_api_token:
                raise ValueError("REPLICATE_API_TOKEN is not set")

            image = generate_image_with_replicate_flux_dev(
                replicate_api_token,
                image_prompt,
                image_aspect_ratio,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generated image:", "done": True},
                }
            )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"![{image_prompt}]({image})  \n**Aspect Ratio:** `{image_aspect_ratio}`  **Prompt:** `{image_prompt}`  \n"
                    },
                }
            )

            return f"""
The image generation completed successfully!

Note that the generated image ALREADY HAS been automatically sent and displayed to the user by the tool.
You don't need to do anything to show the image to the user.

When you answer now - simply tell the user the image was successfully generated.
Answer with text only, NO IMAGES, NO ASPECT RATIO, NO IMAGE PROMPT.
Just tell the user that the image was successfully generated.

At the end of your answer you might ask the user if they want anything changed to the image.
Should they later come back and ask for changes - just generate a new image with potentially modified parameters based on their feedback.

Keep your answer short and concise.
"""

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occurred: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"


def generate_image_with_replicate_flux_dev(
    replicate_api_token: str,
    prompt: str,
    aspect_ratio: ImageAspectRatioType,
) -> str:
    url = "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions"

    headers = {
        "Authorization": f"Bearer {replicate_api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "jpg",
            "safety_tolerance": 6,
            "prompt_upsampling": True,
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    response_json = response.json()
    remote_image_url = response_json.get("output", "")
    # THIS IS A FIX - Check if the output is a list and extract the first element
    if isinstance(remote_image_url, list) and len(remote_image_url) > 0:
        return remote_image_url[0]
    else:
        return remote_image_url
