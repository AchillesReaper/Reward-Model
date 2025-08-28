
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List


class PlannerOutput(BaseModel):
    task: str = Field(..., description="The main task description.")
    steps: List[str] = Field(..., description="A list of steps produced by the planner agent.")

    @property
    def step_count(self) -> int:
        """Returns the number of steps."""
        return len(self.steps)


class ResearcherOutput(BaseModel):
    sub_task: str = Field(..., description="The subtask description.")
    findings: List[str] = Field(..., description="A list of findings produced by the researcher agent.")

    @property
    def finding_count(self) -> int:
        """Returns the number of findings."""
        return len(self.findings)


class CriticOutput(BaseModel):
    task: str = Field(..., description="The main task description.")
    critiques: List[str] = Field(..., description="A list of critiques produced by the critic agent.")

    @property
    def critique_count(self) -> int:
        """Returns the number of critiques."""
        return len(self.critiques)