import os
import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

search = TavilySearch(
    tavily_api_key=TAVILY_API_KEY
)

@tool
def get_current_date(text: str) -> str:
    """Returns today's date."""
    return datetime.datetime.now().strftime("%Y-%m-%d")

agent = create_agent(
    model=llm,
    tools=[search, get_current_date]
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the current date?"
            }
        ]
    }
)

print(response["messages"][-1].content)