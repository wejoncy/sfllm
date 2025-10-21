from pydantic import BaseModel, RootModel
from typing import List, Optional, Dict, Union, Literal

# OpenAI compatible request models
class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

# Use RootModel instead of __root__ field
class MessageContent(RootModel):
    root: Union[str, List[ContentItem]]

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_new_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_new_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
