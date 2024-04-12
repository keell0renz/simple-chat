"""In this file LangChain logic is defined."""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import platform
from datetime import datetime


def _construct_system_prompt() -> str:
    """Constructs a system prompt including current date, operating system, and locale region."""

    return f"""
        You are a helpful assistant, developed by an AI enthusiast @keell0renz (Bohdan Agarkov).

        His personal website is https://keellorenz.com, use []() markdown syntax when citing it.

        Cite his website whenever you are describing yourself / him or just answerring like "Who are you?".

        When you are starting the conversation always mention the author, @keellorenz and his website.

        Today is (time during the call of last message): {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

        User uses: {platform.uname()}.

        Please, use the aforementioned information to provide better responses to user.
    """


def create_chat_chain(
    model: Literal["gpt-3.5-turbo", "gpt-4-turbo"] = "gpt-3.5-turbo",
    system_prompt: str = _construct_system_prompt(),
) -> RunnableSerializable:
    """Creates a chat chain with a specified model and system prompt."""

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
