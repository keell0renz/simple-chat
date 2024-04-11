"""In this file LangChain logic is defined."""

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are helpful assistant.
        """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{prompt}"),
    ]
)

chain = prompt | llm | StrOutputParser()
