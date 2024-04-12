"""Main file, the entry point."""

from typing import Any, Literal, cast, List, Dict, Union
from chain import create_chat_chain as get_chain
import base64

import chainlit as cl
from chainlit.input_widget import Select, TextInput
from langchain_core.messages import MessageLikeRepresentation
from langchain.schema import AIMessage, HumanMessage


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.

    Args:
    - image_path (str): The path to the image file.

    Returns:
    - str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@cl.on_chat_start
async def init():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="OpenAI Model",
                values=["gpt-3.5-turbo", "gpt-4-turbo"],
                initial_index=0,
            )
        ]
    ).send()

    chat_history: list[MessageLikeRepresentation] = []

    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("model", settings["model"])


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]):
    cl.user_session.set("model", settings["model"])


@cl.on_message
async def main(message: cl.Message):
    model: Literal["gpt-3.5-turbo", "gpt-4-turbo"] = cl.user_session.get("model")  # type: ignore
    history: list[MessageLikeRepresentation] = cl.user_session.get("chat_history")  # type: ignore
    response = cl.Message(content="")

    images = [file for file in message.elements if file.mime and "image" in file.mime]

    if images and model not in ["gpt-4-turbo"]:
        response.content = (
            "Cannot process images! Please choose a model which supports images."
        )
        response.author = "System"
        await response.send()
        return None

    content = [{"type": "text", "text": message.content}] + [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(file.path)}"
            },
        }
        for file in images
        if file.path
    ]

    content = cast(List[Union[str, Dict]], content)

    history.append(HumanMessage(content=content))

    await response.send()

    try:
        async for chunk in get_chain(model).astream(
            {"chat_history": history, "prompt": message.content}
        ):
            await response.stream_token(chunk)
    except RuntimeError:
        pass

    history.append(AIMessage(content=response.content))
