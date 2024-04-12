"""Main file, the entry point."""

from chain import create_chat_chain  # pylint: disable=import-error
import chainlit as cl
from langchain_core.messages import MessageLikeRepresentation
from langchain.schema import AIMessage, HumanMessage


@cl.on_chat_start
async def init():
    """Setup code, defines session-dependant variables such as chat history."""

    chat_history: list[MessageLikeRepresentation] = []

    cl.user_session.set("chat_history", chat_history)


@cl.on_message
async def main(message: cl.Message):
    """Processes messages sent by user."""

    chain = create_chat_chain()

    history: list[MessageLikeRepresentation] = cl.user_session.get("chat_history")  # type: ignore

    history.append(HumanMessage(content=message.content))

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(
        {"chat_history": history, "prompt": message.content}
    ):
        await msg.stream_token(chunk)

    history.append(AIMessage(content=msg.content))
