import os
import nest_asyncio
from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from tavily import TavilyClient

set_tracing_disabled(True)
nest_asyncio.apply()
load_dotenv()

google_api = os.getenv("GOOGLE_API_KEY")
tavily_api = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api)

google_client = AsyncOpenAI(
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
  api_key=google_api
)

gemini_model = OpenAIChatCompletionsModel(
  model="gemini-2.5-flash",
  openai_client=google_client
)

@function_tool
def search_internet(query: str) -> str:
  """
  Searches for `query` on the internet

  Args:
    query: The search query

  Returns:
    str: Answer to the user query or whatever found for the query 
  """

  response = tavily_client.search(query)
  return response

search_agent_settings = ModelSettings(
  tool_choice="required"
)

search_agent = Agent(
  name="SearchAgent",
  instructions="You have to search for the user query on the internet",
  model=gemini_model,
  tools=[search_internet],
  model_settings=search_agent_settings
)

def main():
  while True:
    try:
      prompt = input("Enter your query: ")

      if prompt == "quit":
        break

      context = {
        "prompt": prompt
      }

      result = Runner.run_sync(search_agent, prompt, context=context)

      print(result.final_output)
    except Exception as e:
      print(f"An error occurred: {e}")

main()
