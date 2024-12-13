#============================================================
# Imports and environment variables
#============================================================

import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import traceback

# Unicode symbols for test status
PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"

def log_status(message, status=True):
    """Log the status of a test with a checkmark or cross."""
    symbol = PASS_SYMBOL if status else FAIL_SYMBOL
    print(f"{symbol} {message}")

try:
    load_dotenv(dotenv_path="/home/parikh92/sciborg_dev/.env")
    log_status("Environment variables loaded successfully.")
except Exception as e:
    log_status(f"Error loading environment variables: {e}", status=False)
    sys.exit(1)

try:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_key
    log_status("OpenAI API key set successfully.")
except Exception as e:
    log_status(f"Error setting OpenAI API key: {e}", status=False)
    sys.exit(1)

sys.path.insert(1, "/home/parikh92/")

try:
    from langchain_openai import ChatOpenAI
    from sciborg_dev.ai.agents.core import create_linqx_chat_agent
    from sciborg_dev.ai.chains.microservice import module_to_microservice
    from sciborg_dev.ai.chains.workflow import create_workflow_planner_chain
    from sciborg_dev.testing.models.drivers import PubChemCaller
    from sciborg_dev.core.library.base import BaseDriverMicroservice
    log_status("Modules imported successfully.")
except ImportError as e:
    log_status(f"Error importing required modules: {e}", status=False)
    sys.exit(1)

#============================================================
# Building the Pubchem Microservice
#============================================================

file_path = '/home/parikh92/sciborg_dev/ai/agents/driver_pubchem.json'

try:
    pubchem_command_microservice = module_to_microservice(PubChemCaller)
    with open(file_path, 'w') as outfile:
        outfile.write(pubchem_command_microservice.model_dump_json(indent=2))
    log_status("PubChem microservice built and saved successfully.")
except Exception as e:
    log_status(f"Error creating PubChem microservice: {e}", status=False)
    sys.exit(1)

#================================================================
# Planner Chain
#================================================================

try:
    planner = create_workflow_planner_chain(
        llm=ChatOpenAI(model='gpt-4'),
        library=pubchem_command_microservice
    )
    log_status("Workflow planner chain initialized successfully.")
except Exception as e:
    log_status(f"Error initializing workflow planner chain: {e}", status=False)
    sys.exit(1)

try:
    planner_response = planner.invoke(
        {
            "query": "What is the IC50 of 1-[(2S)-2-(dimethylamino)-3-(4-hydroxyphenyl)propyl]-3-[(2S)-1-thiophen-3-ylpropan-2-yl]urea to the Mu opioid receptor, cite a specific assay in your response?"
        }
    )
    log_status("Planner chain query invoked successfully.")
    print(planner_response)
except Exception as e:
    log_status(f"Error invoking planner chain: {e}", status=False)

#============================================================
# Pubchem Agent
#============================================================

try:
    pubchem_agent = create_linqx_chat_agent(
        microservice=pubchem_command_microservice,
        llm=ChatOpenAI(model='gpt-4'),
        human_interaction=False,
        verbose=True
    )
    log_status("PubChem agent created successfully.")
except Exception as e:
    log_status(f"Error creating PubChem agent: {e}", status=False)
    sys.exit(1)

try:
    agent_response = pubchem_agent.invoke(
        {
            "input": "What is the Ki of pzm21 to the Mu opioid receptor, cite a specific assay in your response?"
        }
    )
    log_status("PubChem agent query invoked successfully.")
    print(agent_response)
except Exception as e:
    log_status(f"Error invoking PubChem agent: {e}", status=False)

#============================================================
# RAG Agent + Pubchem Microservice
#============================================================

try:
    rag_agent = create_linqx_chat_agent(
        microservice=pubchem_command_microservice,
        rag_vectordb_path='/home/parikh92/sciborg_dev/embeddings/NIH_docs_embeddings',
        llm=ChatOpenAI(model='gpt-4'),
        human_interaction=False,
        verbose=True
    )
    log_status("RAG agent created successfully.")
except Exception as e:
    log_status(f"Error creating RAG agent: {e}", status=False)
    sys.exit(1)

try:
    rag_response = rag_agent.invoke(
        {
            "input": "How does microwave irradiation influence reaction mechanisms differently compared to conventional heating methods?"
        }
    )
    log_status("RAG agent query invoked successfully.")
    print(rag_response)
except Exception as e:
    log_status(f"Error invoking RAG agent: {e}", status=False)

#============================================================
# Final Output
#============================================================

print("\n🎉 All tests completed! 🎉")