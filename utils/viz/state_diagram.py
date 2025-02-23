import networkx as nx
import matplotlib.pyplot as plt

def visualize_execution_flow(intermediate_steps):
    """
    Visualizes the execution flow of an agent's steps.
    
    Parameters:
    - intermediate_steps: List of tuples containing agent actions and responses.
    """
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add start and end nodes
    G.add_node("START")
    G.add_node("END")

    # Process intermediate steps and add nodes/edges
    previous_node = "START"
    node_labels = {"START": "START", "END": "END"}  # Ensure START and END are labeled
    edge_labels = {}  # Store edge sequence numbers

    for step_number, step in enumerate(intermediate_steps, start=1):
        action = step[0].tool
        input_data = step[0].tool_input
        output_data = step[1]

        # Create node label with truncated output
        node_label = f"{action}\nInput: {input_data}"
        G.add_node(action)
        node_labels[action] = node_label  # Store for later custom labeling

        # Add edge with sequence number
        G.add_edge(previous_node, action)
        edge_labels[(previous_node, action)] = str(step_number)  # Numbering the edges
        previous_node = action

    # Connect last node to END
    G.add_edge(previous_node, "END")
    edge_labels[(previous_node, "END")] = str(len(intermediate_steps) + 1)  # Last step

    # Draw the graph
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
    nx.draw(G, pos, labels={node: "" for node in G.nodes}, with_labels=True, 
            node_size=3500, node_color="lightblue", edge_color="black", 
            font_size=10, font_weight="bold", arrows=True)

    # Add custom detailed labels outside the nodes
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, verticalalignment='bottom', 
                            horizontalalignment='center', bbox=dict(facecolor="white", alpha=0.6))

    # Add edge labels to show execution sequence
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

    # Display the graph
    plt.title("Agent Execution Flow Visualization (Numbered Sequence)")
    plt.show()

def pretty_print_intermediate_steps(intermediate_steps):
    """
    Pretty prints the intermediate steps
    """
    print(f"Total steps: {len(result['intermediate_steps'])}")
    print(f"The Execution flow is as follows:")
    print("-"*50)
    print("START")
    print("-"*50)
    for step in result["intermediate_steps"]:
        print("Agent Action:", step[0].tool)
        print("Input:", step[0].tool_input)
        print("Output:", step[1])
        print("\n" + "-"*50)
    print("END")