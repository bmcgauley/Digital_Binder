# Multi-Agent Personal Assistant (Archon MCP)

## Project Overview
This project implements a modular, extensible personal assistant system using Archon MCP and at least 6 specialized agents. Each agent leverages MCP tools and self-guided workflows to autonomously accomplish a wide range of professional and personal tasks, including email management, calendar updates, research, client analysis, web development, and systems analysis.

## Features
- Email management and actionable extraction
- Calendar updates and scheduling
- Research, paper preparation, and review
- Client business analysis and reporting
- Web development (code, debug, deploy)
- Systems analysis and design
- Extensible agent-based architecture

## Agent Descriptions
- **Email Agent**: Manages email, extracts actionable items, and schedules events.
- **Calendar Agent**: Updates and manages the user's calendar based on email, tasks, and user input.
- **Research & Paper Agent**: Prepares, reviews, and summarizes academic or business papers.
- **Client Analysis Agent**: Analyzes client businesses, generates reports, and provides recommendations.
- **Web Development Agent**: Codes, debugs, and deploys websites for clients.
- **Systems Analysis & Design Agent**: Conducts systems analysis, designs solutions, and reports findings.

## Architecture
- **Archon MCP**: Central orchestrator, receives user tasks, delegates to agents, and manages tool usage.
- **Agent Modules**: Modular, self-contained components with access to relevant MCP tools and APIs.
- **Shared Context**: Agents share context and results via a central state or message bus.

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Ensure Archon MCP and other MCPs are available in your environment or via Docker.

## Usage
Run the main entry point:
```bash
python main.py
```

This will demonstrate basic agent delegation. For full orchestration, integrate with Archon MCP and LangGraph as described in the code comments.

## File Structure
- `/agents/` - Agent implementations
- `/tests/` - Unit/integration tests
- `main.py` - Entry point
- `PLANNING.md` - Architecture/design
- `TASK.md` - Task tracking
- `README.md` - This file

## Testing
- Run all tests: `pytest`

## Extending
- Add new agents in `/agents/`
- Update orchestration logic in `main.py`

---
For more details, see `PLANNING.md`. 