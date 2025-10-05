"""
title: Flux Ultra Replicate Image
description: Generate images with Flux 1.1 Pro Ultra hosted on Replicate.
author: Olof Larsson
author_url: https://olof.tech/conversational-image-generation-tool-for-open-webui/
git_url: https://github.com/oloflarsson/openwebui-flux-ultra-replicate
version: 1.1.2
license: MIT
required_open_webui_version: 0.6.1
"""

from typing import Literal, Optional
import requests
import os
import json
from pydantic import BaseModel, Field


# Image aspect ratio types
ImageAspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]


class Tools:
    """
    OpenWebUI tool for generating images using Flux 1.1 Pro Ultra on Replicate.
    """

    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your Replicate API token"
        )

    def __init__(self):
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
        )
        # Disable automatic citations since we emit our own structured messages
        self.citation = False

    async def generate_image(
        self,
        image_prompt: str,
        image_aspect_ratio: ImageAspectRatioType,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
    ) -> str:
        """
        Generate an image given a detailed text prompt and aspect ratio.

        Guidelines:
        - The image_prompt must be in English and describe the image in rich detail.
        - Include the image type (photo, painting, render, etc.) at the start.
        - Use "16:9" as default aspect ratio if none is implied.
        """

        try:
            metadata_mode = (
                __metadata__.get("mode")
                if __metadata__ and isinstance(__metadata__, dict)
                else None
            )

            # Send progress events only if allowed (not in native mode)
            if metadata_mode != "native" and __event_emitter__:
                for _ in range(2):
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Generating image ...",
                                "done": False,
                            },
                        }
                    )

            replicate_api_token = self.valves.REPLICATE_API_TOKEN
            if not replicate_api_token:
                raise ValueError("REPLICATE_API_TOKEN is not set")

            image_url = generate_image_with_replicate_flux_pro_ultra(
                replicate_api_token,
                image_prompt,
                image_aspect_ratio,
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
                "The image was successfully generated and displayed. "
                "Ask the user if they want any changes or adjustments to the result."
            )

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"An error occurred: {e}",
                            "done": True,
                        },
                    }
                )
            return f"An error occurred while generating the image: {e}"


def generate_image_with_replicate_flux_pro_ultra(
    replicate_api_token: str,
    prompt: str,
    aspect_ratio: ImageAspectRatioType,
) -> str:
    """
    Call Replicate's API to generate an image with Flux 1.1 Pro Ultra.
    """

    url = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro-ultra/predictions"
    headers = {
        "Authorization": f"Bearer {replicate_api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }

    payload = {
        "input": {
            "raw": False,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "jpg",
            "safety_tolerance": 6,
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
