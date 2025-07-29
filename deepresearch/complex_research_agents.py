import os
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI

from agents import Agent, Runner, WebSearchTool, RunConfig, set_default_openai_client, HostedMCPTool

# Use env var for API key and set a long timeout
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600.0)
set_default_openai_client(client)

# Disable tracing for Zero Data Retention (ZDR) Organizations
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Tutorial link: https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents


# ─────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────

CLARIFYING_AGENT_PROMPT =  """
    If the user hasn't specifically asked for research (unlikely), ask them what research they would like you to do.

        GUIDELINES:
        1. **Be concise while gathering all necessary information** Ask 2–3 clarifying questions to gather more context for research.
        - Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner. Use bullet points or numbered lists if appropriate for clarity. Don't ask for unnecessary information, or information that the user has already provided.
        2. **Maintain a Friendly and Non-Condescending Tone**
        - For example, instead of saying “I need a bit more detail on Y,” say, “Could you share more detail on Y?”
        3. **Adhere to Safety Guidelines**
        """

RESEARCH_INSTRUCTION_AGENT_PROMPT = """

        Based on the following guidelines, take the users query, and rewrite it into detailed research instructions. OUTPUT ONLY THE RESEARCH INSTRUCTIONS, NOTHING ELSE. Transfer to the research agent.

        GUIDELINES:
        1. **Maximize Specificity and Detail**
        - Include all known user preferences and explicitly list key attributes or dimensions to consider.
        - It is of utmost importance that all details from the user are included in the expanded prompt.

        2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
        - If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to “no specific constraint.”

        3. **Avoid Unwarranted Assumptions**
        - If the user has not provided a particular detail, do not invent one.
        - Instead, state the lack of specification and guide the deep research model to treat it as flexible or accept all possible options.

        4. **Use the First Person**
        - Phrase the request from the perspective of the user.

        5. **Tables**
        - If you determine that including a table will help illustrate, organize, or enhance the information in your deep research output, you must explicitly request that the deep research model provide them.
        Examples:
        - Product Comparison (Consumer): When comparing different smartphone models, request a table listing each model’s features, price, and consumer ratings side-by-side.
        - Project Tracking (Work): When outlining project deliverables, create a table showing tasks, deadlines, responsible team members, and status updates.
        - Budget Planning (Consumer): When creating a personal or household budget, request a table detailing income sources, monthly expenses, and savings goals.
        Competitor Analysis (Work): When evaluating competitor products, request a table with key metrics—such as market share, pricing, and main differentiators.

        6. **Headers and Formatting**
        - You should include the expected output format in the prompt.
        - If the user is asking for content that would be best returned in a structured format (e.g. a report, plan, etc.), ask the Deep Research model to “Format as a report with the appropriate headers and formatting that ensures clarity and structure.”

        7. **Language**
        - If the user input is in a language other than English, tell the model to respond in this language, unless the user query explicitly asks for the response in a different language.

        8. **Sources**
        - If specific sources should be prioritized, specify them in the prompt.
        - Prioritize Internal Knowledge. Only retrieve a single file once.
        - For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
        - For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
        - If the query is in a specific language, prioritize sources published in that language.

        IMPORTANT: Ensure that the complete payload to this function is valid JSON
        IMPORTANT: SPECIFY REQUIRED OUTPUT LANGUAGE IN THE PROMPT
        """


# ─────────────────────────────────────────────────────────────
# Structured outputs (needed only for Clarifying agent)
# ─────────────────────────────────────────────────────────────
class Clarifications(BaseModel):
    questions: List[str]

# ─────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────
research_agent = Agent(
    name="Research Agent",
    model="o3-deep-research-2025-06-26",
    instructions="Perform deep empirical research based on the user's instructions.",
    tools=[WebSearchTool(),
    ]
)

instruction_agent = Agent(
    name="Research Instruction Agent",
    model="gpt-4o-mini",
    instructions=RESEARCH_INSTRUCTION_AGENT_PROMPT,
    handoffs=[research_agent],
)

clarifying_agent = Agent(
    name="Clarifying Questions Agent",
    model="gpt-4o-mini",
    instructions=CLARIFYING_AGENT_PROMPT,
    output_type=Clarifications,
    handoffs=[instruction_agent],
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Decide whether clarifications are required.\n"
        "• If yes → call transfer_to_clarifying_questions_agent\n"
        "• If no  → call transfer_to_research_instruction_agent\n"
        "Return exactly ONE function-call."
    ),
    handoffs=[clarifying_agent, instruction_agent],
)


# ─────────────────────────────────────────────────────────────
#  Auto-clarify helper
# ─────────────────────────────────────────────────────────────
async def basic_research(
    query: str,
    mock_answers: Optional[Dict[str, str]] = None,
    verbose: bool = False,
):
    print("Running research for query:", query)
    stream = Runner.run_streamed(
        triage_agent,
        query,
        run_config=RunConfig(tracing_disabled=True),
    )

    async for ev in stream.stream_events():
        if isinstance(getattr(ev, "item", None), Clarifications):
            reply = []
            for q in ev.item.questions:
                ans = (mock_answers or {}).get(q, "No preference.")
                reply.append(f"**{q}**\n{ans}")
            stream.send_user_message("\n\n".join(reply))
            continue
        if verbose:
            # from pdb import set_trace; set_trace()
            print(ev.type)

    #return stream.final_output
    return stream



import json

def parse_agent_interaction_flow(stream):
    print("=== Agent Interaction Flow ===")
    count = 1

    for item in stream.new_items:
        # Agent name, fallback if missing
        agent_name = getattr(item.agent, "name", "Unknown Agent") if hasattr(item, "agent") else "Unknown Agent"

        if item.type == "handoff_call_item":
            func_name = getattr(item.raw_item, "name", "Unknown Function")
            print(f"{count}. [{agent_name}] → Handoff Call: {func_name}")
            count += 1

        elif item.type == "handoff_output_item":
            print(f"{count}. [{agent_name}] → Handoff Output")
            count += 1

        elif item.type == "mcp_list_tools_item":
            print(f"{count}. [{agent_name}] → mcp_list_tools_item")
            count += 1

        elif item.type == "reasoning_item":
            print(f"{count}. [{agent_name}] → Reasoning step")
            count += 1

        elif item.type == "tool_call_item":
            tool_name = getattr(item.raw_item, "name", None)

            # Skip tool call if tool_name is missing or empty
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue  # skip silently

            tool_name = tool_name.strip()

            args = getattr(item.raw_item, "arguments", None)
            args_str = ""

            if args:
                try:
                    parsed_args = json.loads(args)
                    if parsed_args:
                        args_str = json.dumps(parsed_args)
                except Exception:
                    if args.strip() and args.strip() != "{}":
                        args_str = args.strip()

            args_display = f" with args {args_str}" if args_str else ""

            print(f"{count}. [{agent_name}] → Tool Call: {tool_name}{args_display}")
            count += 1

        elif item.type == "message_output_item":
            print(f"{count}. [{agent_name}] → Message Output")
            count += 1

        else:
            print(f"{count}. [{agent_name}] → {item.type}")
            count += 1


def print_final_output_citations(stream, preceding_chars=50):
    # Iterate over new_items in reverse to find the last message_output_item(s)
    for item in reversed(stream.new_items):
        if item.type == "message_output_item":
            for content in getattr(item.raw_item, 'content', []):
                if not hasattr(content, 'annotations') or not hasattr(content, 'text'):
                    continue
                text = content.text
                for ann in content.annotations:
                    if getattr(ann, 'type', None) == 'url_citation':
                        title = getattr(ann, 'title', '<no title>')
                        url = getattr(ann, 'url', '<no url>')
                        start = getattr(ann, 'start_index', None)
                        end = getattr(ann, 'end_index', None)

                        if start is not None and end is not None and isinstance(text, str):
                            # Calculate preceding snippet start index safely
                            pre_start = max(0, start - preceding_chars)
                            preceding_text = text[pre_start:start].replace('\n', ' ').strip()
                            excerpt = text[start:end].replace('\n', ' ').strip()
                            print("# --------")
                            print("# MCP CITATION SAMPLE:")
                            print(f"#   Title:       {title}")
                            print(f"#   URL:         {url}")
                            print(f"#   Location:    chars {start}–{end}")
                            print(f"#   Preceding:   '{preceding_text}'")
                            print(f"#   Excerpt:     '{excerpt}'\n")
                        else:
                            # fallback if no indices available
                            print(f"- {title}: {url}")
            break


# Entrypoint: wrap in asyncio.run()
if __name__ == "__main__":
    query = "Research the economic impact of semaglutide on global healthcare systems."
    result = asyncio.run(basic_research(query, verbose=True))
    print("\n=== Final Result ===\n")
    print(result)
    parse_agent_interaction_flow(result)
    print_final_output_citations(result)