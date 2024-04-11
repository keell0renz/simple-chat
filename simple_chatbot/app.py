"""Main file, the entry point."""

import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
            You are helpful assistant.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{prompt}")
])

chain = prompt | llm | StrOutputParser()


@cl.on_chat_start
async def init():
    """Setup code, defines LangChain workflow."""

    chat_history = []

    cl.user_session.set("chat_history", chat_history)

@cl.on_message
async def main(message: cl.Message):
    """Handles messages."""

    chat_history = cl.user_session.get("chat_history")

    chat_history.append(HumanMessage(content=message.content))

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream({"chat_history": chat_history, "prompt": message.content}):
        await msg.stream_token(chunk)

    chat_history.append(AIMessage(content=msg.content))
