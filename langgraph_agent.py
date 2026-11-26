import logging
import math
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_operations import get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("calculator_agent")

llm_client = get_llm_client()

MAX_ABS_VALUE = 1e12  
class CalculatorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def check_number(value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Inputs must be numbers.")
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Inputs must be finite numbers.")
    if abs(value) > MAX_ABS_VALUE:
        raise ValueError(f"Numbers larger than {MAX_ABS_VALUE:g} are not allowed.")


@tool
def add(a: float, b: float) -> float:
    check_number(a)
    check_number(b)
    return a + b


@tool
def sub(a: float, b: float) -> float:
    check_number(a)
    check_number(b)
    return a - b


@tool
def div(a: float, b: float) -> float:
    check_number(a)
    check_number(b)
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


tools = [add, sub, div]
llm_with_tools = llm_client.bind_tools(tools)

SYSTEM_PROMPT = """You are a mathematical assistant.
Use the available tools for every addition, subtraction or division.
Never calculate by yourself.
If a tool returns an error, explain it to the user in simple words.
When you have the final result, give a clear and short answer."""



def llm_node(state: CalculatorState):
    try:
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
    except Exception:
        logger.exception("Model call failed")
        return {
            "messages": [
                AIMessage(content="Sorry, something went wrong. Please try again.")
            ]
        }

    logger.info("Tool calls requested: %s", response.tool_calls)
    return {"messages": [response]}



builder = StateGraph(CalculatorState)

builder.add_node("agent", llm_node)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  

# MemorySaver 
graph = builder.compile(checkpointer=MemorySaver())



def ask_question(query: str, thread_id: str = "user-1") -> str:
    query = query.strip()
    if not query:
        return "Please type a question."

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "recursion_limit": 25,  
                "configurable": {"thread_id": thread_id},
            },
        )
    except GraphRecursionError:
        logger.warning("Recursion limit reached")
        return "I stopped because this took too many steps. Try a simpler question."

    final = result["messages"][-1].content

    if isinstance(final, list):
        final = "\n".join(
            block.get("text", "")
            for block in final
            if isinstance(block, dict) and block.get("text")
        )

    return final or "Sorry, I could not produce an answer."


if __name__ == "__main__":
    print(ask_question("add 10 and 5"))
    # Works because the agent remembers the previous answer:
    print(ask_question("now divide that by 3"))
    # Tests error handling:
    print(ask_question("divide that by 0"))
    # A different thread has its own, separate memory:
    print(ask_question("what was my last result?", thread_id="user-2"))


(
    {
        "messages" :[
            {"role":"usere","content":""}
        ]
    }
)

try:
    print(a + 2)
except TypeError:
    print("typerror ")
    





# from typing import Annotated, TypedDict

# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# from langchain_core.tools import tool
# from langgraph.graph import START, StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition

# from llm_operations import get_llm_client

# llm_client = get_llm_client()



# class CalculatorState(TypedDict):
#     question: str
#     messages: Annotated[list[BaseMessage], add_messages]



# @tool
# def add(a: float, b: float) -> float:
#     """Add two numbers and return a + b."""
#     return a + b


# @tool
# def sub(a: float, b: float) -> float:
#     """Subtract b from a and return a - b."""
#     return a - b


# @tool
# def div(a: float, b: float) -> float | str:
#     """Divide a by b and return a / b."""
#     if b == 0:
#         return "Error: cannot divide by zero."
#     return a / b


# tools = [add, sub, div]
# llm_with_tools = llm_client.bind_tools(tools)

# system_prompt = """You are a mathematical assistant.
# Whenever the user asks for addition, subtraction or division,
# use the available tools. Never calculate by yourself, always use the tools.
# When you have the final result, give a clear and detailed answer."""



# def llm_node(state: CalculatorState):
#     response = llm_with_tools.invoke(
#         [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=f"Question: {state['question']}"),
#         ]
#         + state["messages"]
#     )

#     print("Tool calls requested:", response.tool_calls)

#     return {"messages": [response]}



# builder = StateGraph(CalculatorState)

# builder.add_node("agent", llm_node)
# builder.add_node("tools", ToolNode(tools))

# builder.add_edge(START, "agent")
# builder.add_conditional_edges("agent", tools_condition)

# # The loop: after a tool runs, go back to the model
# builder.add_edge("tools", "agent")

# graph = builder.compile()



# def ask_question(query: str):
#     res = graph.invoke(
#         {"question": query, "messages": []},
#         config={"recursion_limit": 10},  # stop endless loops
#     )

#     # Show every step, so you can see the loop
#     for message in res["messages"]:
#         kind = type(message).__name__
#         print(f"[{kind}] {message.content!r}")

#     final = res["messages"][-1].content

#     # Some models return a list of blocks instead of a string
#     if isinstance(final, list):
#         final = "\n".join(
#             block.get("text", "")
#             for block in final
#             if isinstance(block, dict) and block.get("text")
#         )

#     print("result ------>", final)
#     return final


# if __name__ == "__main__":
#     ask_question("add 2 and 3")
#     ask_question("add 10 and 5, then divide the result by 3, then subtract 2")


# from typing import Annotated, TypedDict

# from langchain_core.messages import (
#     BaseMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )
# from langchain_core.tools import tool
# from langgraph.graph import END, START, StateGraph
# from langgraph.graph.message import add_messages

# from llm_operations import get_llm_client

# llm_client = get_llm_client()



# class CalculatorState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]



# # ---------------------------------------------------------------
# @tool
# def add(a: float, b: float) -> float:
#     """Add two numbers and return a + b."""
#     return a + b


# @tool
# def sub(a: float, b: float) -> float:
#     """Subtract b from a and return a - b."""
#     return a - b


# @tool
# def div(a: float, b: float) -> float | str:
#     """Divide a by b and return a / b."""
#     if b == 0:
#         return "Error: cannot divide by zero."
#     return a / b


# tools = [add, sub, div]
# tools_by_name = {t.name: t for t in tools}

# llm_with_tools = llm_client.bind_tools(tools)

# system_prompt = """You are a mathematical assistant.
# Whenever the user asks for addition, subtraction or division,
# use the available tools. Never calculate by yourself, always use the tools.
# When you have the final result, give a clear and detailed answer."""



# def llm_node(state: CalculatorState):
#     # Send the system prompt + the WHOLE message history,
#     # so the model can see earlier tool results.
#     response = llm_with_tools.invoke(
#         [SystemMessage(content=system_prompt)] + state["messages"]
#     )

#     print("Tool calls requested:", response.tool_calls)

#     # Must return a dict that matches the state key
#     return {"messages": [response]}


# def tool_node(state: CalculatorState):
#     last_msg = state["messages"][-1]  # last MESSAGE, not the whole list
#     outputs = []

#     for call in last_msg.tool_calls:
#         tool_fn = tools_by_name.get(call["name"])

#         if tool_fn is None:
#             result = f"Error: unknown tool {call['name']}"
#         else:
#             try:
#                 result = tool_fn.invoke(call["args"])
#             except Exception as exc:
#                 result = f"Error: {exc}"

#         print(f"TOOL: {call['name']}({call['args']}) -> {result}")

#         outputs.append(
#             ToolMessage(
#                 content=str(result),
#                 tool_call_id=call["id"],
#                 name=call["name"],
#             )
#         )

#     # Must return the results so they are added to the messages
#     return {"messages": outputs}


# def router_tool(state: CalculatorState):
#     last_msg = state["messages"][-1]

#     if last_msg.tool_calls:
#         return "tools"

#     return END



# builder = StateGraph(CalculatorState)

# builder.add_node("agent", llm_node)
# builder.add_node("tools", tool_node)

# builder.add_edge(START, "agent")
# builder.add_conditional_edges("agent", router_tool, ["tools", END])

# # The loop: after a tool runs, go back to the model
# builder.add_edge("tools", "agent")

# graph = builder.compile()



# def ask_question(query: str):
#     res = graph.invoke(
#         {"messages": [HumanMessage(content=query)]},
#         config={"recursion_limit": 10},  # stop endless loops
#     )

#     final = res["messages"][-1].content

#     # Some models return a list of blocks instead of a string
#     if isinstance(final, list):
#         final = "\n".join(
#             block.get("text", "")
#             for block in final
#             if isinstance(block, dict) and block.get("text")
#         )

#     print("result ------>", final)
#     return final


# if __name__ == "__main__":
#     ask_question("add 2 and 3")
#     ask_question("add 10 and 5, then divide the result by 3, then subtract 2")