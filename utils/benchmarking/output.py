from sciborg.utils.benchmarking.base import BaseAgentBenchmarker
from typing import Dict, Type, Any, List
from pydantic import BaseModel, PrivateAttr
import re
import json

class AgentOutputBenchmarker(BaseAgentBenchmarker):
    '''
    Simple class for agentic output benchmarking. Similar to LLM based output benchmarking 
    but applied to an agentic framework. 

    To use, provide a dictionary as desired output with output key matching an agent output key 
    and value matching the expected output value.

    If string is provided as the desired output, the output key is assumed to be 
    "output" and the strings value is compared.

    ### Example Usage
    
    ```py
    # Define or import a funciton which return an AgentExecutor object to specificaitons
    def create_my_agent_executor(**kwargs) -> AgentExecutor:
        # Create and return an agent executor object
        ...

    # Define the benchmark with initial input and desired output
    output_benchmark = AgentOutputBenchmarker(
        executor_fn=create_my_agent_executor,
        executor_kwargs={...}, # kwargs appled to agent executor function
        initial_input="Do something to specification X, return observation of parameter Y as 'observed Y'",
        desired_output="observed 1234"
    )

    # Run the benchmark n times
    output_benchmark.benchmark(10)
    ```
    '''
    desired_output: str | List[str] | Dict[str, str | List[str]]

    _desired_output: Dict[str, List[str]] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
    
        if isinstance(self.desired_output, str):
            self.desired_output = {self._default_output_key: [self.desired_output]}
        elif isinstance(self.desired_output, list):
             self.desired_output = {self._default_output_key: self.desired_output}
        elif isinstance(self.desired_output, dict):
            for key, value in self.desired_output.items():
                if isinstance(value, str):
                    self.desired_output[key] = [value]

        self._desired_output = self.desired_output

    def _cast_agent_output(self, agent_output: Dict[str, Any]):
        '''
        For all keys in the desired output keys, cast their value to string
        '''
        for key in self._desired_output.keys():
            agent_output[key] = str(agent_output[key])
        
        return agent_output

    def _compare_output(self, agent_output: str, desired_output: str) -> bool:
        '''
        Compares string equality
        '''
        return agent_output == desired_output
    
class AgentRegexOutputBenchmarker(AgentOutputBenchmarker):
    '''
    Functions similar to the original AgentOutputBenchmarker but
    attemps to match desired regex pattern(s) to the agents output(s).
    '''

    def _compare_output(self, agent_output: str, desired_output: str) -> bool:
        '''
        Compares regex matching
        '''
        if re.search(desired_output, agent_output):
            return True
        else:
            return False

class AgentJsonOutputBenchmarker(BaseAgentBenchmarker):
    '''
    Class for agentic output benchmarking where the output(s) is expected to be JSON
    formatted. The output(s) will be run against the provided Pydantic Validation schema
    to determine pass/fail.

    This class expects that the agent output is either a dictionary or a JSON formatted string which can 
    be converted to a dictionary by the json loads function. 

    It is reccomended (but not required) that you provide a JSON output parser for the agent to prevent
    incorrect final answer formatting.

    To use, provide a dictionary as desired output with output key matching an agent output key 
    and a schema which is run against the agent output value for that key.

    If a schema is provided as the desired output, the output key is assumed to be 
    "output" and the value is run against the provided schema.

    '''

    desired_output: Type[BaseModel] | List[Type[BaseModel]] | Dict[str, Type[BaseModel] | List[Type[BaseModel]]]

    _desired_output: Dict[str, List[Type[BaseModel]]] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)

        if issubclass(self.desired_output, BaseModel):
            self.desired_output = {self._default_output_key: [self.desired_output]}
        elif isinstance(self.desired_output, list):
             self.desired_output = {self._default_output_key: self.desired_output}
        elif isinstance(self.desired_output, dict):
            for key, value in self.desired_output.items():
                if issubclass(value, BaseModel):
                    self.desired_output[key] = [value]

        self._desired_output = self.desired_output

    def _validate_agent_output(self, agent_output: Dict[str, Any]) -> None:
        '''
        Each output should be a dict or string, strings must be JSON deserializable
        '''
        super()._validate_agent_output(agent_output)

        for key in self._desired_output.keys():
            if not isinstance(agent_output[key], str) and not isinstance(agent_output[key], dict):
                raise TypeError("Fatal error, all agent outputs must be of type str or dict")
            if isinstance(agent_output[key], str):
                # If it is a string, it must be JSON deserializable
                json.loads(agent_output[key])

    def _cast_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Deserialize all JSON strings to Python dict
        '''
        for key in self._desired_output.keys():
             if isinstance(agent_output[key], str):
                 agent_output[key] = json.loads(agent_output[key])

    def _compare_output(self, agent_output: Dict[str, Any], desired_output: Type[BaseModel]) -> bool:
        '''
        Validates the agent output against the desired schema, returns true upon success false upon failure
        '''
        try:
            desired_output(**agent_output)
            return True
        except:
            return False