import os
from dotenv import load_dotenv
from typing import List, TypedDict, Optional

# --- LangChain & LangGraph Imports ---
from langchain_core.prompts import ChatPromptTemplate
# We will use the standard Pydantic parser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langgraph.graph import START, END, StateGraph

# --- Tool & Model Imports ---
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from pydantic import BaseModel, Field

# -- Tool Initialization --
load_dotenv()
hf_api_key = os.getenv("hf_api_key")
search_tool = DuckDuckGoSearchRun()
arxiv_tool = ArxivQueryRun()

# -- LLM Initialization --
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", token=hf_api_key)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
chat_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), temperature=0.6)

# -- Agent State (Using TypedDict) --
class AgentState(TypedDict):
    question: str
    research_data: str
    answer: str

system_prompt_template = """
You are an expert research assistant with advanced skills in analyzing academic content, synthesizing key insights, and presenting findings in a structured, accessible format. Your task is to simulate the use of research tools (e.g., literature databases, APIs, or knowledge repositories) to gather information and organize it into the following sections:

1. Key Topics in the Paper : List the central themes, hypotheses, or research questions addressed in the study.
2. Findings/Key Points : Summarize the most significant results, arguments, or discoveries, prioritizing novelty and impact.
3. Methodology Overview : Briefly explain the approach used (e.g., experimental design, data analysis techniques, theoretical frameworks).
4. Implications : Describe the practical or theoretical relevance of the findings.
5. Limitations : Highlight constraints or gaps acknowledged by the authors or inferred from the work.

Guidelines :
1. Use simple, conversational language but ensure depth and detail (verbose yet clear).
2. Structure responses with headings and bullet points for readability.
3. If information is unclear or unavailable, state: ‘The paper does not explicitly address [specific point]’ or ‘I cannot infer [X] from the provided data.’
4. Prioritize recent studies (within the last 5 years) when simulating tool-based searches unless instructed otherwise.
5. Cite sources metaphorically (e.g., ‘According to PubMed’ or ‘A 2023 study in Nature suggests...’ ) to reflect tool usage, even if hypothetical.
6. If the query is vague, ask for clarification (e.g., ‘Are you focusing on applications in healthcare, AI ethics, or another domain?’ ).

[RESEARCH DATA]
Question: {question}
Data: {research_data}
"""

def format_report_markdown(report_text: str) -> str:
    """Converts the structured text report to markdown format"""
    sections = {
        "Key Topics in the Paper": "### Key Topics\n",
        "Findings/Key Points": "### Findings/Key Points\n",
        "Methodology Overview": "### Methodology\n",
        "Implications": "### Implications\n",
        "Limitations": "### Limitations\n"
    }

    # Convert to markdown format
    for section, header in sections.items():
        report_text = report_text.replace(f"{section} :", header)
        report_text = report_text.replace(f"{section}:", header)

    # Add main header
    return f"## Research Report\n\n{report_text}"

# -- Graph Nodes --

# Node Gather Information
def gather_information_node(state: AgentState) -> dict:
    """Node that runs tools in parallel to gather information."""
    print("---NODE: GATHERING INFORMATION---")
    question = state['question']

    tool_runner = RunnableParallel(
        web_search=search_tool,
        arxiv_search=arxiv_tool
    )
    research_results = tool_runner.invoke(question)

    formatted_data = (
        f"--- Web Search Results ---\n{research_results['web_search']}\n\n"
        f"--- Arxiv Paper Results ---\n{research_results['arxiv_search']}"
    )

    return {"research_data": formatted_data}

# Node Generating Report
def synthesize_report_node(state: AgentState) -> dict:
    """
    Node that calls the LLM to analyze the gathered data and generate
    a structured research report.
    """
    print("-- Generating REPORT --")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "Please generate a comprehensive research report:")
    ])

    synthesis_chain = prompt | chat_model

    # Get the raw text response
    report = synthesis_chain.invoke({
        "question": state['question'],
        "research_data": state['research_data']
    }).content

    return {"answer": report}

# -- Build and Compile the Graph --
workflow = StateGraph(AgentState)
workflow.add_node("gather_information", gather_information_node)
workflow.add_node("synthesize_report", synthesize_report_node)

workflow.set_entry_point("gather_information")
workflow.add_edge("gather_information", "synthesize_report")
workflow.add_edge("synthesize_report", END)

research_graph = workflow.compile()

try:
    research_topic = "What are the latest advancements in Quantum Computing for drug discovery?"
    final_state = research_graph.invoke({"question": research_topic})

    # Format the text report as markdown
    formatted_report = format_report_markdown(final_state['answer'])

    print("  FINAL RESEARCH REPORT  ")

    print(formatted_report)

except Exception as e:
    print(f"\nAn error occurred during the research process: {e}")