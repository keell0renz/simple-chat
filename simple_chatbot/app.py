"""Main file, the entry point."""

import chainlit as cl


@cl.on_message
async def main(message: cl.Message):
    """Handles messages."""

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()
