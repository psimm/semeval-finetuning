from typing import List
from pydantic import BaseModel


class Aspect(BaseModel):
    term: str
    polarity: str


class AbsaAnswer(BaseModel):
    aspects: List[Aspect]
