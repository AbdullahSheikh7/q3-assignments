from py_types import GlobalContext
from models import gemini
from tool_functions import get_order_status
from agents import ModelSettings, RunContextWrapper, Agent, handoff
from guardrails import input_guardrail, output_guardrail

my_model_settings = ModelSettings(
  tool_choice="auto"
)

human_agent = Agent(
  name="HumanAgent",
  instructions="You are a human agent and you resolve the user query, you cannot handoff it further and can't delegate the job, you are the final one to answer the query and you have to resolve the prompt anyway",
  model=gemini,
  handoff_description="A human agent and it should be handoffed the prompt if it is a complex or negative sentiment"
)

def human_agent_handoffed(ctx: RunContextWrapper[GlobalContext]):
  print("Shifted to Human Agent")

human_agent_handoff = handoff(human_agent, on_handoff=human_agent_handoffed)

bot_agent = Agent(
  name="BotAgent",
  instructions="""You are a **helpful customer support** agent. You task is to:
- Answer basic product FAQs
- Fetch order status
- Escalate to a human agent if the query is complex or sentiment is negative

Company details:
- Company Name: TechGear Pro
- Products: Electronics, gadgets, and accessories
- Shipping: 2-5 business days (domestic), 7-14 days (international)
- Return Policy: 30-day return window, items must be unopened
- Payment Methods: Credit/Debit cards, PayPal, Digital Wallets

Common FAQs:
1. How do I track my order?
   - Use order ID to check status through our system
2. What's the return process?
   - Contact support within 30 days
   - Get return authorization
   - Ship item back in original packaging
3. Do you ship internationally?
   - Yes, to most countries with extended delivery time
4. Are products under warranty?
   - Most electronics come with 1-year manufacturer warranty
5. What if my item arrives damaged?
   - Report within 48 hours with photos for immediate replacement""",
  model=gemini,
  handoffs=[human_agent_handoff],
  input_guardrails=[input_guardrail.input_guardrail],
  output_guardrails=[output_guardrail.output_guardrail],
  tools=[get_order_status],
  tool_use_behavior="run_llm_again",
  model_settings=my_model_settings
)
