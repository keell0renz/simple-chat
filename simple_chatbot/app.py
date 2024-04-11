"""Main file, the entry point."""

from chain import chain  # pylint: disable=import-error
import chainlit as cl
from langchain.schema import AIMessage, HumanMessage


@cl.on_chat_start
async def init():
    """Setup code, defines session-dependant variables such as chat history."""

    chat_history = []

    cl.user_session.set("chat_history", chat_history)


@cl.on_message
async def main(message: cl.Message):
    """Processes messages sent by user."""

    chat_history = cl.user_session.get("chat_history")

    chat_history.append(HumanMessage(content=message.content))

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(
        {"chat_history": chat_history, "prompt": message.content}
    ):
        await msg.stream_token(chunk)

    chat_history.append(AIMessage(content=msg.content))
