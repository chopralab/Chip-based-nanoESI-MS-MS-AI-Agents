import operator
import networkx as nx
import matplotlib.pyplot as plt
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# Step 1: Define the Tools (Math Operations)
@tool
def add_numbers(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y

@tool
def subtract_numbers(x: float, y: float) -> float:
    """Subtracts two numbers."""
    return x - y

math_tools = [add_numbers, subtract_numbers]

# Step 2: Define the Agent State (Tracking Execution)
class AgentState(TypedDict):
    input: str  # User input
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    execution_history: Annotated[list[dict], operator.add]  # Track execution

# Step 3: Define the Math Agent (Decision Making)
def run_agent(data):
    """Simulates an agent deciding which math operation to perform."""
    text = data['input']

    if "add" in text:
        action = AgentAction(
            tool="add_numbers",
            tool_input={"x": 3, "y": 7},  # Example fixed values
            log="Adding numbers"
        )
    elif "subtract" in text:
        action = AgentAction(
            tool="subtract_numbers",
            tool_input={"x": 10, "y": 4},
            log="Subtracting numbers"
        )
    else:
        return {"agent_outcome": AgentFinish(output="Invalid operation!", log="Error")}

    # Track agent decision
    if "execution_history" not in data:
        data["execution_history"] = []
    
    data["execution_history"].append({
        "step": len(data["execution_history"]) + 1,
        "node": "agent",
        "action": action.tool,
        "input": action.tool_input,
        "output": None  # To be updated after execution
    })

    return {"agent_outcome": action, "execution_history": data["execution_history"]}

# Step 4: Define Tool Execution
def execute_tools(data):
    """Executes the selected math operation."""
    agent_output = data["agent_outcome"]
    
    if isinstance(agent_output, AgentAction):
        tool_name = agent_output.tool
        tool_input = agent_output.tool_input

        # Execute the tool
        if tool_name == "add_numbers":
            output = add_numbers.invoke(tool_input)
        elif tool_name == "subtract_numbers":
            output = subtract_numbers.invoke(tool_input)
        else:
            output = "Unknown tool"

        # Log execution step
        data["execution_history"].append({
            "step": len(data["execution_history"]) + 1,
            "node": "action",
            "action": tool_name,
            "input": tool_input,
            "output": str(output)
        })

        return {"intermediate_steps": [(agent_output, str(output))], "execution_history": data["execution_history"]}
    
    return {"intermediate_steps": [], "execution_history": data["execution_history"]}

# Step 5: Define Conditional Logic
def should_continue(data):
    """Determines whether to continue or exit."""
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    return "continue"

# Step 6: Define and Compile the Workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Define Execution Flow
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

# Compile into a runnable object
app = workflow.compile()

# Initialize the state with execution tracking
initial_state = AgentState(
    input="add two numbers",
    agent_outcome=None,
    intermediate_steps=[],
    execution_history=[]
)

# Run the workflow
final_state = app.invoke(initial_state)

# Print execution history
import json
print(json.dumps(final_state["execution_history"], indent=4))

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges from execution history
history = final_state["execution_history"]

for i, step in enumerate(history):
    node_label = f"Step {step['step']}\n{step['node']}\nAction: {step['action']}\nOutput: {step['output']}"
    G.add_node(i, label=node_label)
    if i > 0:
        G.add_edge(i - 1, i)  # Connect previous step to current step

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, "label")
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray")
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
plt.title("Math Agent Execution Flow")
plt.show()