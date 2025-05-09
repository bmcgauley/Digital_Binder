from .base import Agent
from typing import Any, Dict
from pydantic import Field

class CalendarAgent(Agent):
    """
    Agent for managing calendar events.
    """
    name: str = Field(default="CalendarAgent")
    description: str = Field(default="Manages calendar events.")

    def run(self, task: str, context: Dict[str, Any]) -> str:
        # Example: Use context to create a calendar event
        email_result = context.get("email_result", None)
        # Reason: This agent can use the result of the email agent if present
        if email_result:
            result = f"Calendar event created based on: {email_result}"
        else:
            result = f"Calendar event created for task: {task}"
        return result 