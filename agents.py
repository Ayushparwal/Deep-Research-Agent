import os
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from linkup import LinkupClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Load .env variables
load_dotenv()


# ✅ Fix for Ollama + LiteLLM
def get_llm_client():
    return LLM(
        model="ollama/tinyllama:1.1b-chat",  # Required format
        base_url="http://localhost:11434",
        api_key="ollama"  # Just a dummy value for LiteLLM
    )


# ✅ LinkUp Tool input schema
class LinkUpSearchInput(BaseModel):
    query: str = Field(description="The search query to perform")
    depth: str = Field(default="standard", description="Search depth: 'standard' or 'deep'")
    output_type: str = Field(default="structured", description="Output format: 'searchResults', 'sourcedAnswer', or 'structured'")


# ✅ LinkUp Tool class
class LinkUpSearchTool(BaseTool):
    name: str = "LinkUp Search"
    description: str = "Search the web for information using LinkUp and return comprehensive results"
    args_schema: Type[BaseModel] = LinkUpSearchInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str, depth: str = "standard", output_type: str = "structured") -> str:
        try:
            print(f"[🔍 LinkUp Search] Searching for: {query}")
            linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

            result = linkup_client.search(
                query=query,
                depth=depth,
                output_type=output_type
            )

            return str(result)
        except Exception as e:
            return f"Error during LinkUp search: {str(e)}"


# ✅ Crew creator with real query passed
def create_research_crew(query: str):
    linkup_tool = LinkUpSearchTool()
    client = get_llm_client()

    web_searcher = Agent(
        role="Web Searcher",
        goal="Find the most relevant information on the web, with source links.",
        backstory="An expert at searching and gathering high-quality online information.",
        verbose=True,
        allow_delegation=True,
        tools=[linkup_tool],
        llm=client,
    )

    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize information from search results.",
        backstory="A critical thinker who transforms raw data into structured insights.",
        verbose=True,
        allow_delegation=True,
        llm=client,
    )

    technical_writer = Agent(
    role="Technical Writer",
    goal="Answer the user's original query in a clear, fact-based, and simple manner using the research analysis. Always include relevant examples and cite sources.",
    backstory="An expert communicator who explains complex topics in simple words for a general audience.",
    verbose=True,
    allow_delegation=False,
    llm=client,
)


    # ✅ Now passing the real query to tool arguments
    search_task = Task(
        description=f"Search for information about: '{query}' using LinkUp.",
        agent=web_searcher,
        expected_output="Raw search results with sources (urls).",
        tools=[linkup_tool],
        tool_choice={
            "name": "LinkUp Search",
            "arguments": {
                "query": query,
                "depth": "standard",
                "output_type": "structured"
            }
        }
    )

    analysis_task = Task(
        description="Analyze the search results, extract important insights, verify facts.",
        agent=research_analyst,
        expected_output="Verified insights with relevant sources.",
        context=[search_task]
    )

    writing_task = Task(
        description="Create a final markdown response based on analysis.",
        agent=technical_writer,
        expected_output="Well-structured answer with citations and clear explanations.",
        context=[analysis_task]
    )

    crew = Crew(
        agents=[web_searcher, research_analyst, technical_writer],
        tasks=[search_task, analysis_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    return crew


# ✅ Final function to run the full process
def run_research(query: str):
    try:
        crew = create_research_crew(query)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        return f"Error during crew execution: {str(e)}"


