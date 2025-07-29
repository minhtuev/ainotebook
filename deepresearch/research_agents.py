import os
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI

from agents import Agent, Runner, WebSearchTool, RunConfig, set_default_openai_client

# Use env var for API key and set a long timeout
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600.0)
set_default_openai_client(client)

# Disable tracing for Zero Data Retention (ZDR) Organizations
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Define the research agent
research_agent = Agent(
    name="Research Agent",
    model="o4-mini-deep-research-2025-06-26",
    tools=[WebSearchTool()],
    instructions="You perform deep empirical research based on the user's question."
)

# Async function to run the research and print streaming progress
async def basic_research(query: str):
    print(f"Researching: {query}")
    result_stream = Runner.run_streamed(research_agent, query)

    async for ev in result_stream.stream_events():
        if ev.type == "agent_updated_stream_event":
            print(f"\n--- switched to agent: {ev.new_agent.name} ---")
            print(f"\n--- RESEARCHING ---")
        elif ev.type == "raw_response_event":
            item = getattr(ev.data, "item", None)
            action = getattr(item, "action", None)
            if getattr(action, "type", None) == "search":
                print(f"[Web search] query={getattr(action, 'query', '')!r}")


    return result_stream.final_output

# Entrypoint: wrap in asyncio.run()
if __name__ == "__main__":
    query = "Research the economic impact of semaglutide on global healthcare systems."
    result = asyncio.run(basic_research(query))
    print("\n=== Final Result ===\n")
    print(result)
