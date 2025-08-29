from typing import Any
from openai import AsyncOpenAI
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,OutputGuardrailTripwireTriggered,
    set_tracing_disabled
)
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel
import os

load_dotenv(find_dotenv(), override=True)
set_tracing_disabled(True)

api_key = os.getenv("OLLAMA_API_KEY")
base_url = "http://localhost:11434/v1"
model_name = "gemma3:1b"

class MathGuardrailOutput(BaseModel):
    is_math: bool
    reason: str

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(openai_client=client, model=model_name)

@input_guardrail
async def math_input_guardrail(
    ctx: RunContextWrapper[MathGuardrailOutput],
    agent: Agent[MathGuardrailOutput],
    input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    math_input_guardrail_agent = Agent[MathGuardrailOutput](
        name="InputGuardrailAgent",
        instructions='''These are for your instructions, it's not the user prompt: "Output `is_math` to `True` only and only if the user prompt is directly related to 'MATHEMATICS' else set `is_math` to `False` and tell the reason why the `is_math` is `True` or `False`"''',
        model=model,
        output_type=MathGuardrailOutput,
    )

    result = await Runner.run(math_input_guardrail_agent, f"User prompt: ```{input_data}```", context=ctx.context)
    final_output = result.final_output

    print(f"\nInput\nIs math: {final_output.is_math}\nReason: {final_output.reason}")

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_math
    )

@output_guardrail
async def math_output_guardrail(
    ctx: RunContextWrapper[MathGuardrailOutput],
    agent: Agent[MathGuardrailOutput],
    input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    math_input_guardrail_agent = Agent[MathGuardrailOutput](
        name="OutputGuardrailAgent",
        instructions='''These are for your instructions, it's not the agent response: "Output `is_math` to `True` only if the response contains MATHEMATICAL content and NO political topics or references to political figures. Set `is_math` to `False` if the response mentions any political content or figures. Provide the reason why `is_math` is `True` or `False`"''',
        model=model,
        output_type=MathGuardrailOutput,
    )

    result = await Runner.run(math_input_guardrail_agent, f"Agent response: ```{input_data}```", context=ctx.context)
    final_output = result.final_output

    print(f"\nOutput\nIs math: {final_output.is_math}\nReason: {final_output.reason}")

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_math
    )

math_agent = Agent(
    name="MathAgent",
    instructions="You are a math agent",
    model=model,
    input_guardrails=[math_input_guardrail],
)

class MathOutput(BaseModel):
    result: int

def main():
    try:
        msg = input("Enter you question: ")
        result = Runner.run_sync(math_agent, msg)
        print(f"\nFinal output: {result.final_output}")

    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as output:
        print(f"Error: {output}")

main()