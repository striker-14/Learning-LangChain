# All agents include a sequence of messages in their state; To invoke the agent, pass a new message along with a thread_id so the agent can persist and resume conversation history. The important concept is that the agent remembers the first message when you ask the second message because both calls use the same thread_id.

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7     # A UUID is a unique identifier.
from langgraph.checkpoint.memory import InMemorySaver   # This provides a place to save the agent's conversation state in memory (RAM).
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}  # generates a unique ID that can be used to identify a conversation.

result = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Pranay"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config,
)

print(result["messages"][-1].content_blocks)