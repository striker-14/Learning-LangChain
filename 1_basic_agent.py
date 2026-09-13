from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""     # That docstring is important because it tells the LLM what the tool does.
    return f"It's always sunny in {city}!"  # F-Strings (Format Strings)

agent = create_agent(
    model = "google_genai:gemini-3.1-flash-lite",
    tools = [get_weather],
    system_prompt = "You are a helpful assistant",
)

result = agent.invoke(  # invoke() runs the agent.
    {"messages": [{"role": "user", "content": "What's the weather in Mumbai"}]}
)

print(result["messages"][-1].content_blocks) # get the last message in the messages list, the last message is the agent's final answer.