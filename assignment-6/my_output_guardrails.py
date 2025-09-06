from pydantic import BaseModel
from my_models import gemini
from agents import RunContextWrapper, output_guardrail, Agent,GuardrailFunctionOutput, Runner

class OutputGuardrail(BaseModel):
  contains_usa_city: bool
  reason: str

@output_guardrail
def usa_output_guardrail(ctx: RunContextWrapper[OutputGuardrail], agent: Agent[OutputGuardrail], output_data: str) -> GuardrailFunctionOutput:
  result = Runner.run_sync(
    Agent(
      name="USAOutputGuardrailAgent",
      model=gemini,
      output_type=OutputGuardrail,
    ),
    f"Does this response contain any USA city: {output_data}",
    context=ctx.context
  )

  return GuardrailFunctionOutput(
    output_info=result.final_output,
    tripwire_triggered=result.final_output.contains_usa_city
  )
