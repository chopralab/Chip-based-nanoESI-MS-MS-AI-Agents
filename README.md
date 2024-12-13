# SciBORG

In the version update branch we use a venv to maintain the new list of packages till the time the entire shift is done. The new list of packages is in the new_requirements.txt file. The new venv is created using the following command:
```shell
python3.10 -m venv env
source env/bin/activate
```

<img src="images/logo.png" alt="SciBORG Banner Image" width="" height="150">

SciBORG is an innovative framework designed for building agents that can rapidly automate scientific discovery. It’s built to be modular, extensible, and easy to integrate with new components and agents, making it suitable for diverse research domains. 

This framework allows for smooth integration of various AI agents, each customizable to specific tasks, making it ideal for research labs, academic institutions, and organizations focusing on scientific discovery. By leveraging SciBORG, teams can automate complex workflows, empowering researchers to focus on innovative thinking and problem-solving.

Task:(some tracking tags go here)

<h3>Key Features</h3>

- Modular Design: Easily add, replace, or upgrade agents as per project needs.
- Extensible Framework: Integrate new components effortlessly.
- Agent Library: Ready-to-use agents like the RAG agent, which supports retrieval-augmented generation for scientific Q&A tasks.
- Benchmarking and Testing: Built-in benchmarking tools to ensure reliability and effectiveness of each agent.

## Quick Start Guide

### Pre-requisites
For compatibility, it’s recommended to use a virtual environment with Python 3.10. Follow these steps to set it up:
```shell
python3.10 -m venv sciborg-env
source sciborg-env/bin/activate
```

### Installation
To install the SciBORG agent, clone the repository and install the dependencies using the following commands:
```shell
git clone https://github.com/chopralab/sciborg_dev.git
cd sciborg_dev
pip install -r requirements.txt
```

### Testing the Installation
Test the installation by running the following command:
```python
import os
os.environ['OPENAI_API_KEY'] = "<YOUR_API_KEY_HERE>"

import sys
sys.path.insert(1, "path to the parent of this folder") 

from langchain_openai import ChatOpenAI
from sciborg_dev.ai.agents.core import create_linqx_chat_agent
from sciborg_dev.ai.chains.microservice import module_to_microservice
from sciborg_dev.ai.chains.workflow import create_workflow_planner_chain, create_workflow_constructor_chain

from sciborg_dev.testing.models.drivers import MicrowaveSynthesizer, MicrowaveSynthesizerObject, PubChemCaller
from sciborg_dev.core.library.base import BaseDriverMicroservice
```
If the above command runs without any errors, the installation was successful and the SciBORG agent is ready to use.

### Directory Structure
If the installation was successful, the directory structure should look like this:
```
sciborg_dev/
  ai/
  core/
  embeddings/
  LCU/
  lcu-remote-client/
  microservices/
  notebooks/
  testing/
  utils/
  README.md
  requirements.txt
```

#### Directory Structure Explanation:

- `ai/`: Contains implementations of various AI agents that drive SciBORG’s functionality, including the RAG agent and other pre-built agents.
- `core/`: The core logic and infrastructure that powers SciBORG, including utilities for agent management, execution, and task orchestration.
- `embeddings/`: Stores models and logic for generating and managing document embeddings. This folder is essential for supporting tasks like document similarity and context retrieval.
- `microservices/`: Houses microservices that enable SciBORG to interact with other applications, databases, and APIs, facilitating the broader applicability of the agents.
- `notebooks/`: Jupyter notebooks providing tutorials, demos, and examples for using SciBORG agents. This folder is highly recommended for new users to understand SciBORG’s capabilities.
- `testing/`: Test cases and scripts to verify the accuracy, reliability, and performance of each component and agent within SciBORG.
- `utils/`: Utility scripts and helper functions that support various tasks across SciBORG, like benchmarking.

## Creating Agents

### Making the microservice for the agents
#### Microwave Synthesizer microservice
```python
file_path = 'path_to_json/driver_MicroSynth.json'

driver_command_microservice = module_to_microservice(MicrowaveSynthesizer)

with open(file_path, 'w') as outfile:
    outfile.write(driver_command_microservice.model_dump_json(indent=2))
```
#### PubChem microservice
```python
file_path = 'path_to_json/driver_pubchem.json'

pubchem_command_microservice = module_to_microservice(PubChemCaller)

with open(file_path, 'w') as outfile:
    outfile.write(pubchem_command_microservice.model_dump_json(indent=2))
```

### RAG Agent

The RAG Agent (Retrieval-Augmented Generation) is used for answering scientific questions based on provided context. To use the RAG agent, run the following command:

```python
rag_agent = create_linqx_chat_agent(
    microservice=driver_command_microservice,
    rag_vectordb_path = '<path>/embeddings/NIH_docs_embeddings',
    llm=ChatOpenAI(model='gpt-4'),
    human_interaction=False,
    verbose=True
)
```

### Pubchem Agent
```python
pubchem_agent = create_linqx_chat_agent(
    microservice=pubchem_command_microservice,
    llm=ChatOpenAI(model='gpt-4'),
    human_interaction=False,
    verbose=True
)
```

### PubChem and RAG Agent
```python
pubchem_and_rag_agent = create_linqx_chat_agent(
    microservice=pubchem_command_microservice,
    rag_vectordb_path = '<path>/embeddings/NIH_docs_embeddings',
    llm=ChatOpenAI(model='gpt-4'),
    human_interaction=False,
    verbose=True
)
```

What do these paramaters mean?
- `microservice`: The microservice that the agent will use to interact with external services or APIs.
- `rag_vectordb_path`: The path to the document embeddings used by the RAG agent for context retrieval.
- `llm`: The language model used by the agent for generating responses. In this case, it’s the GPT-4 model.
- `human_interaction`: A boolean flag to enable or disable human interaction with the agent.

For detailed demonstrations and examples, refer to the Jupyter notebooks `notebooks/SI/SI_traces_01.ipynb` and `notebooks/SI/SI_traces_02.ipynb` in the repository.
Task: link the folder to the notebooks

## Benchmarking the Agents

SciBORG includes benchmarking scripts to evaluate the performance of each agent. There are three ways to benchmark the agents:
- State Based Benchmarking
- Path Based Benchmarking
- RegEx Based Benchmarking

For additional details on benchmarking and usage, refer to the `notebooks/SI/SI_benchmarks_01.ipynb`, `notebooks/SI/SI_benchmarks_02.ipynb` and `notebooks/SI/SI_benchmarks_03.ipynb` notebooks.
Task: link the folder to the notebooks

## Additional Resources (adding soon...)
Explore additional details on extending SciBORG’s functionality with custom integrations:
- Creating Custom Integrations: Learn how to add new agents to SciBORG, configure workflows, and customize agents for specific needs. (add in readme file to help do this)
- Adding Custom Document Embeddings: Understand how to integrate different embeddings for specialized document types or formats. (add in a readme file and a code file to help do this)
