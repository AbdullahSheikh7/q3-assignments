import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")

google_client = AsyncOpenAI(
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
  api_key=google_key
)

gemma = OpenAIChatCompletionsModel(
  model="gemma-3-1b-it",
  openai_client=google_client
)

gemini = OpenAIChatCompletionsModel(
  model="gemini-2.5-flash",
  openai_client=google_client
)