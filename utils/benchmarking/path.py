from typing import Iterable, Tuple, List, Dict, Any, Type
from sciborg_dev.utils.benchmarking.base import BaseAgentBenchmarker

from pydantic import BaseModel, InstanceOf, ValidationError, PrivateAttr

from langchain_core.agents import AgentAction

# Custom type for user input action list
action_list_type = List[str | Tuple[str, Type[BaseModel]]]

class AgentPathBenchmarker(BaseAgentBenchmarker):
    '''
    Class for path based agentic benchmarking.

    On each benchmarking run, this benchmarking class invokes the agent with the provided input, 
    waits until the agent is finished, and then compares its action path (intermediate_steps) to 
    the provided desired action paths.

    If the agents action path matches one of the desired action paths, the run is a success.
    Otherwise the run is a failure.

    You must ensure that the intermediate steps of the agent exector are returned
    in the agent output.
    '''
    desired_output: action_list_type | List[action_list_type] | Dict[str, action_list_type | List[action_list_type]]

    _desired_output: Dict[str, List[action_list_type]] | None = PrivateAttr(default=None)
    _default_output_key: str = PrivateAttr(default='intermediate_steps')

    def model_post_init(self, __context: Any):
        # Assign post init of the superclass
        super().model_post_init(__context)

        # If it an individual output make it a list of one element and assign to default output key
        if isinstance(self.desired_output, list) and all((isinstance(elem, str) or isinstance(elem, tuple) for elem in self.desired_output)):
            self.desired_output = {self._default_output_key: [self.desired_output]}
        # If it is a list, make it a dictionary with the default output key
        elif isinstance(self.desired_output, list):
            self.desired_output = {self._default_output_key: self.desired_output}
        # If it is a dictionary, make sure that all values are lists
        elif isinstance(self.desired_output, dict):
            for key, value in self.desired_output.items():
                if isinstance(value, list) and all((isinstance(elem, str) or isinstance(elem, tuple) for elem in value)):
                    self.desired_output[key] = [value]

        # Assign (should fail validation if something is incorrect)
        self._desired_output = self.desired_output

    def _compare_output(
        self,
        action_path: List[Tuple[str, Dict[str, Dict[str, Any]]]],
        desired_path: action_list_type
    ) -> bool:
        '''
        Compares the action path matches the provided desired path
        '''
        # If there is a length mismatch, it will fail 
        if len(desired_path) != len(action_path):
            return False

        # Determine shorter path to avoid error
        shorter_path_length = len(desired_path) if len(desired_path) <= len(action_path) else len(action_path)

        # Iterate through all actions in the path
        match_list = []
        for i in range(shorter_path_length):
            # If the desired path has a wildcard, the action passes
            if isinstance(desired_path[i], str) and desired_path[i] == "*":
                match_list.append(True)
            # If the desired path only provides a action name, confirm it
            elif isinstance(desired_path[i], str):
                if desired_path[i] != action_path[i][0]:
                    match_list.append(False)
                else:
                    match_list.append(True)
            # If the desired path provides both a name and input, confirm them
            elif isinstance(desired_path[i], tuple):
                if desired_path[i][0] != action_path[i][0]:
                    match_list.append(False)
                else: 
                    # Validate the provided schema if the actions match
                    try:
                        desired_path[i][1](**action_path[i][1])
                        match_list.append(True)
                    except ValidationError as e:
                        match_list.append(False)
            # If there is a typing issue, raise error
            else:
                raise TypeError("Invalid type found in a desired path")

        # If any element is invalid return False, else return True
        return not any(not elem for elem in match_list)
    
    def _validate_agent_output(self, agent_output: Dict[str, Any]) -> None:
        '''
        Validates the agent output path
        '''
        super()._validate_agent_output(agent_output)

        for key in self._desired_output.keys():
            action_path = agent_output[key]
            if not isinstance(action_path, list):
                raise ValueError('action path should be a list')
            if not all(isinstance(elem, tuple) for elem in action_path):
                raise ValueError('Expected intermediate step elements to be tuples')
            if not all(isinstance(elem[0], AgentAction) for elem in action_path):
                raise ValueError('Expected first tuple element to be an AgentAction object')
    
    def _cast_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Casts action path output to standard format for all keys listed in the desired path

        Default is intermediate_steps
        '''
        for key in self._desired_output.keys():
            action_path = agent_output[key]
            # Assign the action path
            action_path = [(action[0].tool, action[0].tool_input) for action in action_path]
            # Standardize inputs to dictionaries
            action_path = [(action[0], {'input': action[1]}) if isinstance(action[1], str) else action for action in action_path]
            agent_output[key] = action_path
        
        return agent_output
    
    def _format_agent_output(self, agent_output: List[Tuple[str, Dict]]) -> str:
        str_agent_output = ''
        for tool_name, tool_input in agent_output:
            str_agent_output += f'{tool_name}: {tool_input}\n'
        return f'- Agent action path:\n{str_agent_output}'