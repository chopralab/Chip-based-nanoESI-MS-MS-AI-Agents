from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import List, Dict, Any, TypedDict
import tensorflow as tf
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Import your react_agent modules
from react_agent.state import State, InputState  # if needed elsewhere
from react_agent.utils2.extract_medInfo import extraction_medInfo_chain
from react_agent.utils2.recommendation_formatter import formatting_medInfo_chain
from react_agent.tools2.extract_history_tools import extract_history_tools
from react_agent.tools2.local_data_analysis import plot_and_save_icp_chart
from react_agent.utils2.send_email import send_email
from react_agent.utils2.send_sms import send_sms_with_media
from react_agent.utils2.paths import sender_email, sender_email_password
from langchain.agents import AgentExecutor

# Import message types from langchain.schema
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# ----------------- Define a message-based state ------------------
class StudioAgentState(TypedDict):
    messages: List[BaseMessage]
    agent_state: Dict[str, Any]

# ----------------- Functional Implementations ------------------

def summarize_patient_history() -> str:
    """Run the extraction and formatting chains to summarize patient history."""
    medical_history_path = "/Users/atharvaparikh/Desktop/Purdue/ChopraLab/langgraph_implementations/path/to/your/app/src/react_agent/medical_history/Medical History 1.pdf"
    agent_executor = AgentExecutor(agent=extraction_medInfo_chain, tools=extract_history_tools, verbose=True)
    result = agent_executor.invoke({
        "question": f"What is the medical intervention for this patient: doc path = {medical_history_path} ?",
    })
    agent_executor2 = AgentExecutor(agent=formatting_medInfo_chain, tools=[], verbose=True)
    result_formatted = agent_executor2.invoke({
        "question": f"Format this recommendation: {result['output']}",
    })
    return result_formatted['output']

def predict_ICP_values(input_values: List[List[float]]) -> List[float]:
    """Load the TensorFlow model and predict ICP values from input data."""
    model_dir = '/Users/atharvaparikh/Desktop/Purdue/ChopraLab/langgraph_implementations/path/to/your/app/src/react_agent/model/nICP_cur_LSTM_model_2'
    model = tf.saved_model.load(model_dir)
    infer = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(input_values, dtype=tf.float32)
    input_key = list(infer.structured_input_signature[1].keys())[0]
    output = infer(**{input_key: input_tensor})
    output_key = list(output.keys())[0]
    result = output[output_key].numpy()
    return result

def need_to_alert(predicted_values: List[List[float]], threshold: float) -> List[float] | None:
    """Determine if any predicted value exceeds the threshold, and sample the data if so."""
    for timestep in predicted_values:
        for value in timestep:
            if value > threshold:
                return timestep[::4]  # sample every 4th value
    return None

# ----------------- LangGraph Studio Nodes ------------------

def load_excel_node(state: StudioAgentState) -> StudioAgentState:
    """Load input data from a predefined Excel file."""
    excel_path = '/Users/atharvaparikh/Desktop/Purdue/ChopraLab/langgraph_implementations/path/to/your/app/src/react_agent/tests/Data for Demo/demo_input_3.xlsx'
    try:
        df = pd.read_excel(excel_path)
        # Assume first row (columns 2 onward) contains MAP and second row nICP values.
        df_MAP = df.iloc[:1, 2:]
        df_nicp = df.iloc[1:, 2:]
        array1 = df_MAP.values
        array2 = df_nicp.values
        combined_array = np.stack([array1, array2], axis=-1)  # Shape (1, 6000, 2)
        state["messages"].append(AIMessage(content=f"Excel file loaded successfully with shape: {combined_array.shape}"))
        state["agent_state"]["input_values"] = combined_array
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error loading Excel file: {e}"))
    return state

def predict_icp_node(state: StudioAgentState) -> StudioAgentState:
    """Run the ICP prediction function using loaded input values."""
    input_values = state["agent_state"].get("input_values")
    if input_values is None:
        state["messages"].append(AIMessage(content="Input values not found."))
        return state
    try:
        predicted = predict_ICP_values(input_values)
        state["messages"].append(AIMessage(content=f"ICP values predicted. Output shape: {predicted.shape}. Max value: {np.max(predicted)}"))
        state["agent_state"]["predicted"] = predicted
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error predicting ICP: {e}"))
    return state

def check_alert_node(state: StudioAgentState) -> StudioAgentState:
    """Check if the predicted ICP values trigger an alert condition."""
    predicted = state["agent_state"].get("predicted")
    if predicted is None:
        state["messages"].append(AIMessage(content="Predicted values not found."))
        return state
    try:
        reshaped = np.reshape(predicted, (1, 1000))
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error reshaping predicted values: {e}"))
        return state
    threshold = state["agent_state"].get("threshold", 7.0)
    alert_data = need_to_alert(reshaped, threshold)
    if alert_data is not None:
        state["messages"].append(AIMessage(content=f"Alert condition met. {len(alert_data)} alert values sampled."))
    else:
        state["messages"].append(AIMessage(content="No alert condition met."))
    state["agent_state"]["alert_data"] = alert_data
    return state

def summarize_node(state: StudioAgentState) -> StudioAgentState:
    """Summarize patient history using the extraction and formatting chains."""
    try:
        summary = summarize_patient_history()
        state["messages"].append(AIMessage(content=f"Patient history summarized: {summary[:100]}..."))
        state["agent_state"]["summary"] = summary
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error summarizing patient history: {e}"))
    return state

def plot_node(state: StudioAgentState) -> StudioAgentState:
    """Generate a chart from alert data and save the plot."""
    alert_data = state["agent_state"].get("alert_data")
    if alert_data is None:
        state["messages"].append(AIMessage(content="No alert data available for plotting."))
        return state
    try:
        threshold = state["agent_state"].get("threshold", 7.0)
        plot_path = plot_and_save_icp_chart(alert_data, threshold)
        state["messages"].append(AIMessage(content=f"Chart generated at: {plot_path}"))
        state["agent_state"]["plot_path"] = plot_path
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error generating chart: {e}"))
    return state

def notify_node(state: StudioAgentState) -> StudioAgentState:
    """Send notifications (email and SMS) based on the summarized history and generated plot."""
    summary = state["agent_state"].get("summary")
    plot_path = state["agent_state"].get("plot_path")
    patientID = state["agent_state"].get("patientID", "P123")
    subject = f"Patient Alert : {patientID}"
    try:
        send_email(
            receiver_email="parikh92@purdue.edu",
            subject=subject,
            body=summary,
            sender_email=sender_email,
            password=sender_email_password,
            filename=plot_path
        )
        state["messages"].append(AIMessage(content="Notification sent. The patient needs immediate attention."))
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Notification error: {e}"))
    state["agent_state"]["status"] = "alert_sent"
    return state

# ----------------- Conditional Logic ------------------

def should_alert(state: StudioAgentState) -> str:
    return "alert_needed" if state["agent_state"].get("alert_data") is not None else "no_alert"

# ----------------- Build the LangGraph Studio Graph ------------------

builder = StateGraph(StudioAgentState, input=StudioAgentState)

builder.add_node("load_data", RunnableLambda(load_excel_node))
builder.add_node("predict_icp", RunnableLambda(predict_icp_node))
builder.add_node("check_alert", RunnableLambda(check_alert_node))
builder.add_node("summarize", RunnableLambda(summarize_node))
builder.add_node("plot", RunnableLambda(plot_node))
builder.add_node("notify", RunnableLambda(notify_node))

builder.set_entry_point("load_data")
builder.add_edge("load_data", "predict_icp")
builder.add_edge("predict_icp", "check_alert")
builder.add_conditional_edges("check_alert", n, {
    "alert_needed": "summarize",
    "no_alert": "__end__"
})
builder.add_edge("summarize", "plot")
builder.add_edge("plot", "notify")
builder.add_edge("notify", "__end__")

patient_monitor_graph = builder.compile()
patient_monitor_graph.name = "HEMAI Studio Agent"

# ----------------- Run Example ------------------

if __name__ == "__main__":
    # The initial state for LangGraph Studio.
    # In the Studio UI, you would supply a similar JSON structure.
    initial_state: StudioAgentState = {
         "messages": [HumanMessage(content="analyze this patient")],
         "agent_state": {
             "patientID": "P123",
             "threshold": 7.0
         }
    }
    result = patient_monitor_graph.invoke(initial_state)
    
    # Print final state and message log for debugging.
    print("Final Graph State:")
    print(result)
    print("\nMessage Log:")
    for msg in result["messages"]:
        print(f"- {msg.content}")


# { "patientID": "P123", "threshold": 7.0 }