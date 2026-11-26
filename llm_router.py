
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from llm_operations import get_llm_client
from llm_operations import generate_llm_resposne
from sql_agent_tools import ask_sql_question


class OrchestratorState(TypedDict, total=False):
    question: str
    route: str
    answer: str


llm = get_llm_client()


def classify_question(state: OrchestratorState):
    question = state["question"]

    prompt = f"""
    Decide which workflow should answer the question.

    Return only one label:
    RAG - questions about document content, policies, explanations.
    SQL - questions requiring structured data, calculations,
          counts, totals, or database records.

    Question: {question}

    Label:
    """

    response = llm.invoke(prompt)
    route = response.content.strip().upper()

    if route not in {"RAG", "SQL"}:
        route = "RAG"

    return {"route": route}


def choose_workflow(
    state: OrchestratorState,
) -> Literal["rag", "sql"]:
    if state["route"] == "SQL":
        return "sql"

    return "rag"


def run_rag(state: OrchestratorState):
    answer = generate_llm_resposne(state["question"])
    return {"answer": str(answer)}


def run_sql(state: OrchestratorState):
    answer = ask_sql_question(state["question"])
    return {"answer": str(answer)}


builder = StateGraph(OrchestratorState)

builder.add_node("classify", classify_question)
builder.add_node("rag", run_rag)
builder.add_node("sql", run_sql)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    choose_workflow,
    {
        "rag": "rag",
        "sql": "sql",
    },
)

builder.add_edge("rag", END)
builder.add_edge("sql", END)

orchestrator = builder.compile()


def ask_question(question: str):
    result = orchestrator.invoke({"question": question})
    return result["answer"]
