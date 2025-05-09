from pydantic import BaseModel
from typing import Any, Dict

class Agent(BaseModel):
    """
    Base class for all agents in the personal assistant system.
    """
    name: str
    description: str

    def run(self, task: str, context: Dict[str, Any]) -> Any:
        """
        Execute the agent's main function for a given task.

        Args:
            task (str): The task description or command.
            context (Dict[str, Any]): Shared context or state.

        Returns:
            Any: The result of the agent's execution.
        """
        raise NotImplementedError("Each agent must implement the run method.") 