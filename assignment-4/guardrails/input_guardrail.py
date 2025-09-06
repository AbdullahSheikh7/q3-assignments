from pydantic import BaseModel
from models import gemini
from agents import RunContextWrapper, input_guardrail, Agent,GuardrailFunctionOutput, Runner, TResponseInputItem
from typing import List

class InputGuardrail(BaseModel):
  is_offensive_or_negative: bool
  reason: str

@input_guardrail
def input_guardrail(ctx: RunContextWrapper[InputGuardrail], agent: Agent[InputGuardrail], input_data: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="GeneralAgent",
      model=gemini,
      output_type=InputGuardrail
    ),
    f"Is this prompt offensive or negative: {input_data}",
    context=ctx.context
  )

  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_offensive_or_negative
  )
