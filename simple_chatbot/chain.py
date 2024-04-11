"""In this file LangChain logic is defined."""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_chat_chain(
    model: Literal["gpt-3.5-turbo", "gpt-4-turbo-0125"] = "gpt-3.5-turbo",
    system_prompt: str = "You are helpful assistant.",
) -> RunnableSerializable:
    """Creates a chat chain with a specified model and system prompt.

    Args:
        model (Literal["gpt-3.5-turbo", "gpt-4-turbo-0125"]): The model to use for the chat.
        system_prompt (str): The initial system prompt for the chat.

    Returns:
        A chat chain configured with the specified model and system prompt.
    """

    llm = ChatOpenAI(model=model, streaming=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{prompt}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain
