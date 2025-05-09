import pytest
from agents import (
    EmailAgent,
    CalendarAgent,
    ResearchPaperAgent,
    ClientAnalysisAgent,
    WebDevelopmentAgent,
    SystemsAnalysisAgent,
)

def test_email_agent():
    agent = EmailAgent()
    result = agent.run("Check email", {})
    assert "EmailAgent" in result

def test_calendar_agent():
    agent = CalendarAgent()
    result = agent.run("Update calendar", {})
    assert "CalendarAgent" in result

def test_research_paper_agent():
    agent = ResearchPaperAgent()
    result = agent.run("Prepare research paper", {})
    assert "ResearchPaperAgent" in result

def test_client_analysis_agent():
    agent = ClientAnalysisAgent()
    result = agent.run("Analyze client", {})
    assert "ClientAnalysisAgent" in result

def test_web_development_agent():
    agent = WebDevelopmentAgent()
    result = agent.run("Debug website", {})
    assert "WebDevelopmentAgent" in result

def test_systems_analysis_agent():
    agent = SystemsAnalysisAgent()
    result = agent.run("Conduct systems analysis", {})
    assert "SystemsAnalysisAgent" in result 