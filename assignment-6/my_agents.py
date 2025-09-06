from my_models import gemini
from agents import ModelSettings, Agent
from my_tools import web_search
from my_input_guardrails import flight_input_guardrail, hotel_input_guardrail, weather_input_guardrail

general_agent_settings = ModelSettings(
  tool_choice="required"
)

flight_agent = Agent(
  name="FlightAgent",
  instructions="""
You are a helpful flight information assistant.
Your job is to provide accurate flight schedules, fares, and booking information based on user queries.
You can answer questions about flight availability, prices, routes, and travel requirements.
If you need to use tools or external data to answer, do so efficiently.
Always clarify departure/arrival locations and dates if the user query is ambiguous.
Be concise, factual, and friendly in your responses.
""",
  model=gemini,
  tools=[web_search],
  tool_use_behavior="run_llm_again",
  model_settings=general_agent_settings,
  input_guardrails=[flight_input_guardrail]
)

hotel_agent = Agent(
  name="HotelAgent", 
  instructions="""
You are a helpful hotel booking assistant.
Your job is to provide accurate hotel availability, rates, and booking information based on user queries.
You can answer questions about room types, amenities, pricing, and booking policies.
If you need to use tools or external data to answer, do so efficiently.
Always clarify location and dates if the user query is ambiguous.
Be concise, factual, and friendly in your responses.
""",
  model=gemini,
  tools=[web_search],
  tool_use_behavior="run_llm_again",
  model_settings=general_agent_settings,
  input_guardrails=[hotel_input_guardrail]
)

weather_agent = Agent(
  name="WeatherAgent",
  instructions="""
You are a helpful and knowledgeable weather assistant. 
Your job is to provide accurate, up-to-date weather information, forecasts, and advice based on user queries. 
You can answer questions about current weather conditions, forecasts for specific locations, severe weather alerts, and general weather-related advice. 
If you need to use tools or external data to answer, do so efficiently. 
Always clarify location and time if the user query is ambiguous. 
Be concise, factual, and friendly in your responses.
""",
  model=gemini,
  tools=[web_search],
  tool_use_behavior="run_llm_again",
  model_settings=general_agent_settings,
  input_guardrails=[weather_input_guardrail]
)

triage_agent = Agent(
  name="TriageAgent",
  instructions="""
You are a triage agent responsible for routing user queries to specialized agents.
Route travel queries about:
- Flights to the FlightAgent
- Hotels to the HotelAgent  
- Weather to the WeatherAgent
""",
  model=gemini,
  handoffs=[flight_agent, hotel_agent, weather_agent],
  model_settings=general_agent_settings
)
