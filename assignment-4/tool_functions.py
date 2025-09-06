from models import gemini
from agents import Agent
from py_types import GlobalContext
from agents import function_tool, RunContextWrapper, Runner, Agent

def is_order_status_enabled(ctx: RunContextWrapper[GlobalContext], agent: Agent[GlobalContext]) -> bool:
  prompt = ctx.context["prompt"]

  result = Runner.run_sync(
    Agent(
      name="GeneralAgent",
      model=gemini,
      output_type=bool,
    ),
    f"Is this prompt related to order status query: {prompt}",
    context=ctx.context
  )

  return result

def function_failed(ctx: RunContextWrapper[GlobalContext], error: Exception):
  print(f"Tool failed to retrieve the order information of the provided id: {error["id"]}")
  return { "message": f"{error["id"]} not found", "status": 404, "id": error["id"] }

@function_tool("OrderStatus", is_enabled=is_order_status_enabled, failure_error_function=function_failed)
def get_order_status(order_id: str) -> str:
  """
  Query the status of an order based on the provided order ID.

  Args:
    ctx: The runtime context wrapper.
    args: JSON string containing the order_id parameter.

  Returns:
    str: Order status message including shipping, delivery, or processing information.
  """

  if order_id == "456":
    return "Order #456: Shipped - Arriving tomorrow"
  elif order_id == "7898":
    return "Order #7898: Processing - Will ship in 2 days"
  elif order_id == "00256":
    return "Order #00256: Delivered on March 15, 2024"
  elif order_id == "1234":
    return "Order #1234: Cancelled - Refund processed"
  elif order_id == "9999":
    return "Order #9999: On hold - Payment verification required"
  elif order_id == "3333":
    return "Order #3333: In transit - Delayed due to weather"
  else:
    raise ValueError({ "message": f"{order_id} not found", "status": 404, "id": order_id })

