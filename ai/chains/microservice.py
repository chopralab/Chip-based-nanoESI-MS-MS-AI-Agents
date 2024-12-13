from tqdm import tqdm
from importlib import import_module
from inspect import getmembers, isfunction, isclass, ismethod, signature, getdoc, getmodule, getsource
from types import ModuleType, FunctionType
from pydantic import BaseModel as BaseModelV2
import json
from langchain_core.language_models import BaseLanguageModel
# from langchain.pydantic_v1 import BaseModel as BaseModelV1
from uuid import UUID, uuid4
from sciborg_dev.core.command.base import BaseDriverCommand, BaseInfoCommand
from sciborg_dev.core.library.base import BaseDriverMicroservice
from sciborg_dev.ai.chains.core import create_linqx_command_parser

def function_to_driver_command(
    func: FunctionType,
    microservice: str, 
    uuid: UUID,
    llm: BaseLanguageModel | None = None, # Defaults to GPT 3.5
) -> BaseDriverCommand:
    '''
    Converts a generic Python function into a BaseDriverCommand

    Parameters
    ```
    func: FunctionType # The python fuction to convert
    microserivce: str # The microservice the function belongs to
    uuid: UUID # The uuid of the microservice
    llm: BaseLanguageModel | None = None # Custom LLM for parsing (default to GPT 3.5 turbo)
    ```
    '''
    # Define function input query
    query = """
    Create a command from the below information. 
    Do not include any parameters that are not in the function signature.
    If the function signature denotes that it returns something, check to docstring to find the return signature.

    Function name: {name}
    Microservice: {microservice}
    UUID: {uuid}
    Function Signature: {signature}
    Docstring: {doc}
    """.format(
        name=func.__name__,
        microservice=microservice,
        uuid=str(uuid),
        signature=str(signature(func)),
        doc=getdoc(func)
    )

    # Use LLM to format
    parser = create_linqx_command_parser(llm)
    output = parser.invoke(query)

    print(output)
    # Build info command from this
    # info_command = BaseInfoCommand(**output['text'])
    print(type(output))
    # print(**output)
    info_command = BaseInfoCommand(**output)

    # Create and return driver command
    return BaseDriverCommand(
        name=func.__name__,
        microservice=microservice,
        uuid=uuid,
        desc=info_command.desc,
        parameters=info_command.parameters,
        has_return=info_command.has_return,
        return_signature=info_command.return_signature,
        fn=func
    )

def module_to_microservice(
    module: str | ModuleType,
    package: str | None = None,
    microservice: str | None = None,
    uuid: str | UUID | None = None,
    llm: BaseLanguageModel | None = None,
) -> BaseDriverMicroservice:
    """
    Converts a Python module into a BaseDriverMicroservice.

    Parameters
    ```
    module: str | ModuleType # The module or name of the module
    package: str | None # The package of the module
    microservice: str | None # The name of the microservice (defaults to module name)
    uuid: str | UUID | None # The UUID of the microservice
    ```
    """
    if isinstance(module, str):
        module = import_module(module, package)

    if microservice is None:
        microservice = module.__name__.split('.')[-1]

    if uuid is None:
        uuid = uuid4()
    elif isinstance(uuid, str):
        uuid = UUID(uuid)

    # Get all functions in the module and process them with a progress bar
    functions = [
        (name, func) for name, func in getmembers(
            module,
            lambda o: isfunction(o) and getmodule(o) == module and not o.__name__.startswith('_')
        )
    ]

    driver_command_set = {}
    with tqdm(total=len(functions), desc="Processing functions", unit="function") as pbar:
        for name, func in functions:
            try:
                driver_command_set[name] = function_to_driver_command(func, microservice, uuid, llm)
            except Exception as e:
                print(f"Error processing function {name}: {e}")
            pbar.update(1)

    return BaseDriverMicroservice(
        name=microservice,
        uuid=uuid,
        desc=getdoc(module) if getdoc(module) is not None else "",
        commands=driver_command_set
    )

def object_to_microservice(
    object: object,
    microservice: str | None = None,
    uuid: str | UUID | None = None,
    llm: BaseLanguageModel | None = None,
) -> BaseDriverMicroservice:
    '''
    Converts an object to a microservice.
    '''
    if isinstance(object, BaseModelV2):
        raise NotImplementedError("Version of Langchain used does not support Pydantic V2 schema, this will be updated!")
    
    if isinstance(object, BaseModelV2):
        pass

    # If the microservice is none, set to object class name
    if microservice is None:
        microservice = object.__class__.__name__

    # If there is no UUID assigned, assign one
    if uuid is None:
        uuid = uuid4()
    elif isinstance(uuid, str):
        uuid = UUID(uuid)

    driver_command_set = {
        name : function_to_driver_command(func, microservice, uuid, llm)
        for name, func in getmembers(
            object,
            lambda o: ismethod(o) and getmodule(o) == getmodule(object) and not o.__name__.startswith('_')
        )
    }

    # Return the microservice
    return BaseDriverMicroservice(
        name=microservice,
        uuid=uuid,
        desc=getdoc(object) if getdoc(object) is not None else "",
        commands=driver_command_set,
        microservice_object=object
    )