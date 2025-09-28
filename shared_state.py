from typing import Any

from pydantic import BaseModel


class AppState(BaseModel):
    model: Any
    classes: list[str]


state = AppState(model="", classes=[])
