import os
from dotenv import load_dotenv
from agents import function_tool
from tavily import TavilyClient

load_dotenv()

tavily_api = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api)

@function_tool
def web_search(query: str) -> str:
  """
  Searches for `query` on the internet

  Args:
    query: The search query

  Returns:
    str: Answer to the user query or whatever found for the query 
  """

  response = tavily_client.search(query)
  return response

