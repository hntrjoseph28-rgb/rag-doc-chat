from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage


# Load your API key from the .env file
load_dotenv()

# Choose the model
model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# List of messages to send
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage("What is a data pipeline? Explain in one sentence.")
]

# Call the model
response = model.invoke(messages)

# Print the results
print(type(response))
print(response.content)

print("\n--- Streaming ---")

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
    

print()

print("\n--- Batching ---")

batch_inputs = [
    [HumanMessage(content="What is LangChain in one sentence?")],
    [HumanMessage(content="what is a vector database in one sentence?")],
    [HumanMessage(content="What is RAG in one sentence?")]
]

results = model.batch(batch_inputs)

for result in results:
    print("-", result.content)
