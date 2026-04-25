from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

history = ChatMessageHistory()

history.add_user_message("My name is Joseph.")
response1 = model.invoke(history.messages)
history.add_ai_message(response1.content)
print(response1.content)

history.add_user_message("What is my name?")
response2 = model.invoke(history.messages)
print(response2.content)