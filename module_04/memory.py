from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

history = []

user_message_1 = HumanMessage(content="My name is Joseph.")
history.append(user_message_1)

response1 = model.invoke(history)
history.append(AIMessage(content=response1.content))

print(response1.content)

user_message_2 = HumanMessage(content="What is my name?")
history.append(user_message_2)

response2 = model.invoke(history)

print(response2.content)