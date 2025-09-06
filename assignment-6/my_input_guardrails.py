from pydantic import BaseModel
from my_models import gemini
from agents import RunContextWrapper, input_guardrail, Agent,GuardrailFunctionOutput, Runner, TResponseInputItem
from typing import List

class MainInputGuardrail(BaseModel):
  is_not_allowed: bool
  reason: str

class FlightInputGuardrail(BaseModel):
  is_not_allowed: bool
  reason: str

class HotelInputGuardrail(BaseModel):
  is_not_allowed: bool
  reason: str

class WeatherInputGuardrail(BaseModel):
  is_not_allowed: bool
  reason: str

@input_guardrail
def flight_input_guardrail(ctx: RunContextWrapper[FlightInputGuardrail], agent: Agent[FlightInputGuardrail], input_data: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="FlightInputGuardrailAgent",
      model=gemini,
      output_type=FlightInputGuardrail
    ),
    f"Do not allow if this is not a query about flights, flight schedules, fares, booking, or travel requirements, or if it is a query about India: {input_data}",
    context=ctx.context
  )
  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_not_allowed
  )

@input_guardrail
def hotel_input_guardrail(ctx: RunContextWrapper[HotelInputGuardrail], agent: Agent[HotelInputGuardrail], input_data: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="HotelInputGuardrailAgent",
      model=gemini,
      output_type=HotelInputGuardrail
    ),
    f"Do not allow if this is not a query about hotels, hotel availability, rates, booking, room types, amenities, or booking policies, or if it is a query about India: {input_data}",
    context=ctx.context
  )
  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_not_allowed
  )

@input_guardrail
def weather_input_guardrail(ctx: RunContextWrapper[WeatherInputGuardrail], agent: Agent[WeatherInputGuardrail], input_data: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="WeatherInputGuardrailAgent",
      model=gemini,
      output_type=WeatherInputGuardrail
    ),
    f"Do not allow if this is not a query about weather, forecasts, current conditions, severe weather alerts, or weather advice, or if it is a query about India: {input_data}",
    context=ctx.context
  )
  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_not_allowed
  )

@input_guardrail
def main_input_guardrail(ctx: RunContextWrapper[MainInputGuardrail], agent: Agent[MainInputGuardrail], input_data: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="MainInputGuardrailAgent",
      model=gemini,
      output_type=MainInputGuardrail
    ),
    f"Do not allow if this is not a query for flight, hotel or weather or if it is a query about India: {input_data}",
    context=ctx.context
  )

  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_not_allowed
  )
