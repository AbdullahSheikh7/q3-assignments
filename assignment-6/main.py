import nest_asyncio
from my_agents import triage_agent
from agents import RunConfig, Runner, set_tracing_disabled, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from my_input_guardrails import main_input_guardrail
from my_output_guardrails import usa_output_guardrail

nest_asyncio.apply()
set_tracing_disabled(True)

my_runner_config = RunConfig(
  input_guardrails=[main_input_guardrail],
  output_guardrails=[usa_output_guardrail]
)

def main():
  while True:
    try:
      prompt = input("Enter your question: ")

      if prompt == "quit":
        break

      result = Runner.run_sync(triage_agent, prompt, run_config=my_runner_config)
      print(f"Final answer: {result.final_output}")
    except InputGuardrailTripwireTriggered:
      print(f"Please ask a query that is related to flight, hotel or weather and not related to India")
    except OutputGuardrailTripwireTriggered:
      print(f"Responding about any USA city by the agent is prohibited")
    except Exception as e:
      print(f"An error occurred: {e}")

main()
