from langgraph.graph import StateGraph, END
from typing import List, TypedDict
from pydantic import BaseModel, Field
from main import llm, logger
import pprint
import os


class GraphState(TypedDict):
    user_prompt: str
    plan: str
    files: List[str]
    codebase: dict
    output_directory: str


class ArchitectNodeOutput(BaseModel):
    """Always use this tool to structure your response to the user."""

    plan: str = Field(
        description="The detailed plan for the software architecture in 2-3 sentences, highly intuitive."
    )
    files: List[str] = Field(
        description="List of files to be created for the architecture e.g. ['app/main.py','models/jobs.py']."
    )


def architect_node(state: GraphState) -> GraphState:
    response = llm.with_structured_output(ArchitectNodeOutput).invoke(
        state["user_prompt"]
    )
    logger.info("Architect generated a plan and file list.")
    print(f"\nLLM Response: {response}")

    updated_state: GraphState = {
        "plan": response.plan,
        "files": response.files,
    }

    return updated_state


def coder_node(state: GraphState) -> GraphState:
    codebase = {}
    for file in state["files"]:
        logger.info(f"Generating code for: {file}")
        prompt = f"""
            You are a senior software developer. You are building a project based on this overall plan: '{state["plan"]}'.
            Your current task is to write the complete, simple, optimized and clean code for the file: '{file}'.
            Write ONLY the code for this file. Do not add any explanations, comments, or markdown formatting like ```python."""
        response = llm.invoke(prompt)
        code = response.content
        codebase[file] = code
        logger.info(f"Code generated for {file}.")
    state["codebase"] = codebase
    return state


def file_writer_node(state: GraphState) -> GraphState:
    output_dir = state["output_directory"]
    codebase = state["codebase"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path, code in codebase.items():
        full_path = os.path.join(output_dir, file_path)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory) and directory != "":
            os.makedirs(directory)

        logger.info(f"Writing file: {full_path}")
        with open(full_path, "w") as f:
            f.write(code)

    logger.info("Project successfully generated!")
    return state

    # initial_state: GraphState = {
    #     "user_prompt": "build me a calculator app in python",
    #     "plan": "",
    #     "files": [],
    # }
    # new_state = architect_node(initial_state)
    # new_state = coder_node(new_state)
    # new_state["output_directory"] = "generated_calculator_app"
    # new_state = file_writer_node(new_state)
    # pprint.pprint(new_state)


workflow = StateGraph(GraphState)

workflow.add_node("architect", architect_node)
workflow.add_node("coder", coder_node)
workflow.add_node("file_writer", file_writer_node)

workflow.set_entry_point("architect")
workflow.add_edge("architect", "coder")
workflow.add_edge("coder", "file_writer")
workflow.add_edge("file_writer", END)

app = workflow.compile()
logger.info("Graph compiled.")

if __name__ == "__main__":
    user_prompt = "Create a simple Flask web server that has one route, '/', which returns 'Hello, World!'."

    initial_state = {
        "user_prompt": user_prompt,
        "output_directory": "generated_flask_app",
    }

    print("Starting Knit.dev agent with Groq...")
    final_state = app.invoke(initial_state)
