"""Main file, the entry point."""

from typing import Any, Literal
from chain import create_chat_chain as get_chain

import chainlit as cl
from chainlit.input_widget import Select, TextInput
from langchain_core.messages import MessageLikeRepresentation
from langchain.schema import AIMessage, HumanMessage


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

    history.append(HumanMessage(content=message.content))

    await response.send()

    try:
        async for chunk in get_chain(model).astream(
            {"chat_history": history, "prompt": message.content}
        ):
            await response.stream_token(chunk)
    except RuntimeError:
        pass

    history.append(AIMessage(content=response.content))
