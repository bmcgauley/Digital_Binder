from agents import (
    EmailAgent,
    CalendarAgent,
    ResearchPaperAgent,
    ClientAnalysisAgent,
    WebDevelopmentAgent,
    SystemsAnalysisAgent,
)
from typing import Dict, Any
from pydantic import BaseModel, Field
# from langgraph import Graph (to be used for orchestration)
import requests
from fastapi import FastAPI, Form, Body, Request
from fastapi.responses import HTMLResponse
import json
import html

class TaskRequest(BaseModel):
    """
    Pydantic model for incoming task requests.
    """
    task: str
    context: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str

# Agent registry for easy lookup
AGENTS = {
    "email": EmailAgent(),
    "calendar": CalendarAgent(),
    "research": ResearchPaperAgent(),
    "client_analysis": ClientAnalysisAgent(),
    "webdev": WebDevelopmentAgent(),
    "systems": SystemsAnalysisAgent(),
}

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

def llm_route_task(task: str) -> str:
    prompt = f"Given the following task, which agent should handle it? Agents: email, calendar, research, client_analysis, webdev, systems. Task: '{task}'. Respond with only the agent name."
    agent = ollama_generate(prompt).strip().lower()
    return agent if agent in AGENTS else None

def delegate_task(task: str, context: Dict[str, Any] = {}) -> Any:
    agent_key = llm_route_task(task)
    if agent_key:
        return AGENTS[agent_key].run(task, context)
    else:
        return f"No suitable agent found for task: {task}"

def orchestrate(task, context):
    # Example: If task involves both email and calendar, call both agents
    if "email" in task and "calendar" in task:
        email_result = AGENTS["email"].run(task, context)
        context["email_result"] = email_result
        calendar_result = AGENTS["calendar"].run(task, context)
        return f"Email: {email_result}\nCalendar: {calendar_result}"
    # Otherwise, route as before
    return delegate_task(task, context)

app = FastAPI()

@app.post("/chat", response_class=HTMLResponse)
async def chat_endpoint(message: str = Form(...)):
    response = ollama_generate(message, model="phi")
    return f"""
    <html>
        <body>
            <h1>Multi-Agent Assistant Chat</h1>
            <form action="/chat" method="post">
                <input name="message" type="text" style="width:300px"/>
                <input type="submit"/>
            </form>
            <p><b>You:</b> {message}</p>
            <p><b>Assistant:</b> {response}</p>
            <p>Use /docs for API testing.</p>
        </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h1>Multi-Agent Assistant Chat</h1>
            <form action="/chat" method="post">
                <input name="message" type="text" style="width:300px"/>
                <input type="submit" value="Chat"/>
            </form>
            <form action="/orchestrate" method="post">
                <input name="task" type="text" style="width:300px"/>
                <input type="submit" value="Orchestrate"/>
            </form>
            <p>Use /docs for API testing.</p>
        </body>
    </html>
    """

@app.post("/api/chat")
def chat_api(request: ChatRequest):
    response = ollama_generate(request.message, model="phi")
    return {"response": response}

@app.post("/orchestrate")
def orchestrate_endpoint(request: TaskRequest = Body(...)):
    """
    Orchestrate multi-agent workflows. If the task involves both email and calendar, call both agents and pass results between them.
    Otherwise, use the default delegate_task logic.
    """
    task = request.task
    context = request.context or {}
    task_lower = task.lower()
    results = {}
    trace = []

    if "email" in task_lower and "calendar" in task_lower:
        trace.append({"agent": "email", "input_context": context.copy()})
        email_result = AGENTS["email"].run(task, context)
        context["email_result"] = email_result
        trace[-1]["output"] = email_result
        trace[-1]["output_context"] = context.copy()

        trace.append({"agent": "calendar", "input_context": context.copy()})
        calendar_result = AGENTS["calendar"].run(task, context)
        context["calendar_result"] = calendar_result
        trace[-1]["output"] = calendar_result
        trace[-1]["output_context"] = context.copy()

        results["email"] = email_result
        results["calendar"] = calendar_result
        results["summary"] = "Email Agent and Calendar Agent were both called."
        results["trace"] = trace
        return results
    # Example: If task involves both client analysis and webdev
    elif "client" in task_lower and "web" in task_lower:
        client_result = AGENTS["client_analysis"].run(task, context)
        context["client_result"] = client_result
        webdev_result = AGENTS["webdev"].run(task, context)
        context["webdev_result"] = webdev_result
        results["client_analysis"] = client_result
        results["webdev"] = webdev_result
        results["summary"] = f"Client Analysis Agent and Web Development Agent were both called."
        return results
    # Otherwise, use default routing
    else:
        result = delegate_task(task, context)
        return {"result": result, "summary": "Default agent routing used."}

@app.post("/orchestrate", response_class=HTMLResponse)
async def orchestrate_form(request: Request):
    form = await request.form()
    task = form.get("task")
    context = {}
    from pydantic import parse_obj_as
    orchestrate_result = orchestrate_endpoint(parse_obj_as(TaskRequest, {"task": task, "context": context}))
    # Pretty-print the result as JSON, escape for HTML
    pretty_result = html.escape(json.dumps(orchestrate_result, indent=2))
    # Optionally, extract summary for emphasis
    summary = orchestrate_result.get("summary", "")
    trace = orchestrate_result.get("trace", [])
    return f"""
    <html>
        <body>
            <h1>Multi-Agent Assistant Orchestration</h1>
            <form action="/chat" method="post">
                <input name="message" type="text" style="width:300px"/>
                <input type="submit" value="Chat"/>
            </form>
            <form action="/orchestrate" method="post">
                <input name="task" type="text" style="width:300px"/>
                <input type="submit" value="Orchestrate"/>
            </form>
            <h2>Orchestration Result:</h2>
            <pre style="background:#f4f4f4;padding:10px;border-radius:5px;">{pretty_result}</pre>
            <p><b>Summary:</b> <span style="color:green;">{html.escape(summary)}</span></p>
            <h2>Agent Trace:</h2>
            <pre style="background:#f4f4f4;padding:10px;border-radius:5px;">{html.escape(json.dumps(trace, indent=2))}</pre>
            <p>Use /docs for API testing.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    print("Welcome to your Multi-Agent Assistant! Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Add user input to history
        history.append({"role": "user", "content": user_input})
        # Compose prompt with history for context (simple version)
        prompt = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])
        prompt += "\nAssistant:"
        # Get response from Ollama (phi)
        response = ollama_generate(prompt, model="phi")
        print(f"Assistant: {response.strip()}")
        # Add assistant response to history
        history.append({"role": "assistant", "content": response.strip()})
