from typing import Any
from openai import AsyncOpenAI
from agents import (
  Agent,
  RunContextWrapper,
  Runner,
  OpenAIChatCompletionsModel,
  set_tracing_disabled
)
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel
import os
import nest_asyncio

nest_asyncio.apply()

load_dotenv(find_dotenv(), override=True)
set_tracing_disabled(True)

api_key = os.getenv("GOOGLE_SECRET_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = "gemini-2.5-flash"

hotel_instructions = [
  """
  You are helpful hotel customer care assistant. hotel total room 200.
    - Hotel name is Hotel Sannata.
    - Hotel Owner name is Mr. Ratan Lal
    - 20 rooms not available for public, Its for special guest.
  """,
  
  """
  You are friendly hotel customer service representative. hotel capacity 150 rooms.
    - Hotel name is Mountain View Resort
    - Hotel Owner name is Mrs. Sarah Johnson
    - 15 rooms reserved for VIP guests and corporate clients
    - Specializing in mountain tourism and adventure sports
  """,
  
  """
  You are professional hotel concierge. hotel features 300 rooms.
    - Hotel name is Ocean Paradise Beach Resort
    - Hotel Owner name is Mr. James Wilson
    - 25 rooms exclusively for honeymoon couples and celebrities
    - Direct beach access and water sports facilities
  """
]

class HotelContext(BaseModel):
  which_hotel: int

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(openai_client=client, model=model_name)

def dynamic_instruction(ctx: RunContextWrapper[HotelContext], agent: Agent):
  return hotel_instructions[int(ctx.context["which_hotel"])]

hotel_assistant = Agent(
  name="Hotel Customer care",
  instructions=dynamic_instruction,
  model=model
)

best_hotel_agent = Agent[HotelContext](
  name="BestHotelChooserAgent",
  instructions=f"You task is to understand user query and tell which hotel will be the best for them from the following array and then just return the element index: ```{hotel_instructions}```",
  model=model,
  output_type=HotelContext
)

# We can also use handsoff for this function but the assignment was to include dynamic instructions to agent

def main():
  prompt = input("Enter your message: ")

  hotel = Runner.run_sync(best_hotel_agent, f"Choose a hotel, Here is user query: `{prompt}`")

  result = Runner.run_sync(hotel_assistant, prompt, context={ "which_hotel": hotel.final_output.which_hotel })
  print(result.final_output)

if __name__ == "__main__":
  main()
