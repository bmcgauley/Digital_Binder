# Multi-Agent Personal Assistant System

An advanced personal assistant system that uses multiple specialized agents to handle a wide range of tasks. The system is built to be self-guided and can leverage various Model Context Protocol (MCP) tools to complete tasks efficiently.

## System Architecture

The system consists of a central coordinator and 7 specialized agents that work together:

1. **Email Manager** (EmailAgent): Handles email-related tasks including checking emails, sending emails, and organizing inbox.

2. **Calendar Manager** (CalendarAgent): Manages calendar-related tasks including scheduling meetings, setting reminders, and checking availability.

3. **Document Specialist** (DocumentAgent): Prepares and reviews papers and documents, formats content, and conducts research for written materials.

4. **Business Analyst** (BusinessAgent): Analyzes client businesses, performs market research, analyzes financials, and develops business strategies.

5. **Web Developer** (DevelopmentAgent): Codes and debugs websites and applications, optimizes performance, and handles API integrations.

6. **Systems Specialist** (SystemsAgent): Performs systems analysis and design, gathers requirements, and develops technical architectures.

7. **Research Specialist** (ResearchAgent): Gathers information, performs research, fact-checks claims, and analyzes data.

## MCP Integration

Each agent is designed to use various Model Context Protocol (MCP) tools to complete their tasks:

- The system uses MCPs like OpenAI, Google APIs, document processing tools, code analyzers, and more
- Each agent is specialized to use the appropriate MCPs for their domain
- Agents can coordinate through the SystemCoordinator to provide comprehensive solutions

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm (v6+)

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/personal-assistant-system.git
cd personal-assistant-system
```

2. Install the dependencies:
```
npm install
```

3. Run the application:
```
node src/app.js
```

## Usage

### API Endpoints

The system provides a simple REST API:

- **POST /task**: Submit a new task to the system
  ```json
  {
    "description": "Check my email and update calendar based on any meeting requests",
    "requirements": "Optional details about specific requirements",
    "deadline": "2025-05-10T15:00:00Z"
  }
  ```
  
- **GET /task/:taskId**: Check the status of a task
- **GET /agents**: List all available agents and their capabilities

### Example Tasks

The system can handle a wide variety of tasks, including:

- "Check my email for any urgent messages"
- "Schedule a meeting with the product team for next week"
- "Prepare a business report about the renewable energy market"
- "Debug the login functionality on our website"
- "Design a system architecture for our new e-commerce platform"
- "Review this technical paper for accuracy and clarity"
- "Research the latest trends in artificial intelligence"

## Extensibility

The system is designed to be easily extensible:

- Add new agents by extending the BaseAgent class
- Register new MCPs with existing agents
- Enhance the coordinator's task distribution logic

## For Academic Purposes

This project demonstrates the implementation of a multi-agent system that uses Model Context Protocols (MCPs) to handle various tasks. It showcases how different specialized agents can work together under a central coordinator to provide a comprehensive personal assistant system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.