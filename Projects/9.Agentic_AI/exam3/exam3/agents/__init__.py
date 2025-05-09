from .base import Agent
from .email_agent import EmailAgent
from .calendar_agent import CalendarAgent
from .research_paper_agent import ResearchPaperAgent
from .client_analysis_agent import ClientAnalysisAgent
from .web_development_agent import WebDevelopmentAgent
from .systems_analysis_agent import SystemsAnalysisAgent
from typing import Any, Dict

def ollama_generate(prompt, model="phi", stream=False):
    """
    Send a prompt to the local Ollama server and return the response.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    if stream:
        # Handle streaming if needed
        return response.iter_lines()
    return response.json()["response"]

