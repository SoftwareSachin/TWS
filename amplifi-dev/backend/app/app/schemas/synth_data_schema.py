from typing import List

from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    answer: str


class SyntheticFactoid(BaseModel):
    Context: str | List[str]
    QA: QAPair
