from pydantic import BaseModel
from models import gemini
from agents import RunContextWrapper, output_guardrail, Agent,GuardrailFunctionOutput, Runner

class OutputGuardrail(BaseModel):
  is_offensive_or_negative: bool
  reason: str

@output_guardrail
def output_guardrail(ctx: RunContextWrapper[OutputGuardrail], agent: Agent[OutputGuardrail], output_data: str) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="GeneralAgent",
      model=gemini,
      output_type=OutputGuardrail,
    ),
    f"Is this response offensive or negative: {output_data}",
    context=ctx.context
  )

  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.is_offensive_or_negative
  )
