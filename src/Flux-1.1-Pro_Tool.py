"""
title: Flux DEV on Replicate.com
description: Generate images with Flux PRO hosted on Replicate.com.
author: LiuC0j
author_url: https://github.com/LiuC0j
git_url: https://github.com/LiuC0j/flux-dev-replicate
version: 1.1.1
license: MIT
required_open_webui_version: 0.6.1
"""

from typing import Literal, Optional
import requests
import os
import json
from pydantic import BaseModel, Field


# Supported aspect ratios
ImageAspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]


class Tools:
    """
    OpenWebUI tool for generating images using Flux PRO on Replicate.com.
    """

    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="",
            description="Your Replicate API token from replicate.com",
        )

    def __init__(self):
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
        )
        # Disable automatic citation generation (we handle events manually)
        self.citation = False

    async def generate_image(
        self,
        image_prompt: str,
        image_aspect_ratio: ImageAspectRatioType,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
    ) -> str:
        """
        Generate an image based on a descriptive English prompt and an aspect ratio.

        Prompt rules:
        - The prompt must be in English and include the image type (photo, painting, render, etc.).
        - It should be highly descriptive (3+ sentences).
        - Default aspect ratio: "16:9" unless otherwise implied.
        """

        try:
            metadata_mode = (
                __metadata__.get("mode")
                if __metadata__ and isinstance(__metadata__, dict)
                else None
            )

            # Emit progress events only in non-native mode
            if metadata_mode != "native" and __event_emitter__:
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

            image_url = generate_image_with_replicate_flux_pro(
                replicate_api_token, image_prompt, image_aspect_ratio
            )

            if metadata_mode != "native" and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generated image", "done": True},
                    }
                )

                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": (
                                f"![{image_prompt}]({image_url})  \n"
                                f"**Aspect Ratio:** `{image_aspect_ratio}`  "
                                f"**Prompt:** `{image_prompt}`  \n"
                            )
                        },
                    }
                )

            return (
                "The image was successfully generated and displayed to the user. "
                "You may now ask if they would like any modifications or changes."
            )

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"An error occurred: {e}", "done": True},
                    }
                )
            return f"An error occurred while generating the image: {e}"


def generate_image_with_replicate_flux_pro(
    replicate_api_token: str,
    prompt: str,
    aspect_ratio: ImageAspectRatioType,
) -> str:
    """
    Call Replicate's API to generate an image with Flux PRO.
    """

    url = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"

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
            "safety_tolerance": 2,
            "prompt_upsampling": True,
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    response_json = response.json()

    # Handle both string and list outputs from Replicate
    output = response_json.get("output", "")
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    return output
