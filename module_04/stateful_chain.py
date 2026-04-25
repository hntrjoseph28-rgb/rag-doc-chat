from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model | parser

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

response1 = chain_with_history.invoke(
    {"question": "My name is Joseph."},
    config={"configurable": {"session_id": "session_1"}}
)
print("\n", response1)

response2 = chain_with_history.invoke(
    {"question": "What is my name?"},
    config={"configurable": {"session_id": "session_1"}}
)
print("\n", response2)


# Different session

response3 = chain_with_history.invoke(
    {"question": "What is my name?"},
    config={"configurable": {"session_id": "session_2"}}
)
print("\n", response3)
