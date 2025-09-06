from pydantic import BaseModel

class GlobalContext(BaseModel):
  prompt: str