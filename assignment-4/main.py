import asyncio
import nest_asyncio
from typing import Optional, List, Any
from my_agents import bot_agent
from py_types import GlobalContext
from agents import Runner, set_tracing_disabled, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper, Agent, TResponseInputItem, ModelResponse, RunHooks, Tool

nest_asyncio.apply()

set_tracing_disabled(True)

class MyLogger(RunHooks):
  async def on_llm_start(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext], system_prompt: Optional[str], input_items: List[TResponseInputItem]) -> None:
    print("Message sent to LLM...")

  async def on_llm_end(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext], response: ModelResponse) -> None:
    print("Recieved response from LLM...")

  async def on_agent_start(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext]) -> None:
    print(f"{agent.name} woke up...")

  async def on_agent_end(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext], output: Any) -> None:
    print(f"{agent.name} returned the final output...")

  async def on_handoff(context: RunContextWrapper[GlobalContext], from_agent: Agent[GlobalContext], to_agent: Agent[GlobalContext]) -> None:
    print(f"Handoffing from {from_agent.name} to {to_agent.name}")
    print(to_agent.instructions)

  async def on_tool_start(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext], tool: Tool) -> None:
    print(f"{tool.name} tool called...")

  async def on_tool_end(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext], tool: Tool, result: str) -> None:
    print(f"{tool.name} tool returned output...")

def main():
  while True:
    try:
      prompt = input("Enter your question: ")

      if prompt == "quit":
        break

      ctx: GlobalContext = {
        "prompt": prompt
      }

      result = Runner.run_sync(bot_agent, prompt, context=ctx, hooks=MyLogger)
      print(f"Final answer: {result.final_output}")
    except InputGuardrailTripwireTriggered:
      print(f"Please ask an appropriate question")
    except OutputGuardrailTripwireTriggered:
      print(f"The agent's response was inappropriate, we're sorry")
    except Exception as e:
      print(f"An error occurred: {e}")

asyncio.run(main())
