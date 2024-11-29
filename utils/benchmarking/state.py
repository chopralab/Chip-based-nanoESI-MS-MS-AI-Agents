from typing import Any, Callable, Type, List, Dict
from sciborg_dev.utils.benchmarking.base import BaseAgentBenchmarker

from pydantic import BaseModel, ValidationError, PrivateAttr
import json

class AgentStateBenchmarker(BaseAgentBenchmarker):
    '''
    Class for state based agentic benchmarking. 

    On each benchmarking run, the system is confirmed to be in the 
    intial state prior to agentic operation, the agent runs the input
    and operates on the system, and the
    '''
    # Public Attributes
    initial_state: Type[BaseModel]
    desired_output: Type[BaseModel] | List[Type[BaseModel]]
    current_state: Callable[..., dict]

    _desired_output: Dict[str, List[Type[BaseModel]]]

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)        
        
        # If it is multiple state schema
        if isinstance(self.desired_output, list):
            self._desired_output = {self._default_output_key: self.desired_output}
        # If it is a single state schema
        elif issubclass(self.desired_output, BaseModel):
            self._desired_output = {self._default_output_key: [self.desired_output]}

        # The desired output dict must have 1 key and it must be the default key, we only validate one state at a time
        if len(self._desired_output.keys()) != 1 or self._default_output_key not in self._desired_output.keys():
            raise KeyError(
            f"""Fatal error - desired output must only contain key: {self._default_output_key}\n
            Desired output keys: {self._desired_output.keys()}"""
            )

    def _format_agent_output(self, agent_output: Any) -> str:
        '''
        Formats the system state in JSON format
        '''
        return f'- System State: \n{json.dumps(self.current_state(), indent=2)}'.strip()

    def _cast_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        '''
        We dont care about the agent output in state based benchmarking but we do need it 
        to have the default key so that the benchmark is run. That is done here.
        '''
        if self._default_output_key not in agent_output.keys():
            agent_output[self._default_output_key] = None
        return agent_output

    def _compare_output(self, agent_output: Dict[str, Any], desired_output: Type[BaseModel]) -> bool:
        '''
        Validates the current state againts the schema, returns True if success False if fail
        '''
        try:
            desired_output(**self.current_state())
            return True
        except:
            return False        

    def _validate_initial_state(self) -> bool:
        '''
        Validates if the system is in it's initial state. Prints
        out errors upon fail with verbose mode set to True.

        ### Return
        False if the current state fails validation agains the final state model,
        True otherwise
        '''
        try:
            self.initial_state(**self.current_state())
        except ValidationError as e:
            self._verbose_log(f'Initial State Mismatch \nError: {e}')
            return False
        return True

    def _benchmark_loop(self) -> bool:
        '''
        Modified benchmarking loop to reset the system and validate the initial state
        '''
        self._validate_initial_state()

        return super()._benchmark_loop()