from typing import Dict, Any, Callable, List
from pydantic import BaseModel, PrivateAttr, ConfigDict
from langchain.agents import AgentExecutor

from langchain.memory import CombinedMemory
from langchain_core.memory import BaseMemory
from langchain_core.agents import AgentAction
import json

from tqdm import tqdm_notebook
from tqdm import tqdm

class BaseAgentBenchmarker(BaseModel):
    '''
    Base class for agentic benchmarking. This class CANNOT be used in its base from.

    When extending this class, update the `_validate_run()` method to the success 
    criteria of your choice in order to make a custom benchmarking class
    '''
    # Config
    model_config = ConfigDict(validate_assignment=True)

    # Public Attributes
    executor_fn: Callable[..., AgentExecutor]
    executor_kwargs: Dict[str, Any]
    initial_input: str
    verbose: bool = True
    notebook: bool = False
    desired_output: Any | List[Any] | Dict[str, Any | List[Any]] = None
    reset_system: Callable | None = None
    validate_reset: bool = None
    
    # Private Attributes
    _success_iter: int = PrivateAttr(default=0)
    _fail_iter: int = PrivateAttr(default=0)
    _default_output_key: str = PrivateAttr(default='output')
    _desired_output: Dict[str, List[Any]] | None = PrivateAttr(default=None)
    _tqdm: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any):
        '''
        Post init assignments
        '''
        if self.notebook:
            self._tqdm = tqdm_notebook
        else:
            self._tqdm = tqdm      
    
    @property
    def success_iter(self):
        '''
        Returns the current number of successful iterations
        '''
        return self._success_iter
    
    @property
    def fail_iter(self):
        '''
        Returns the current number of failed iterations
        '''
        return self._fail_iter
    
    @property
    def total_iter(self):
        '''
        Returns the total number of iterations run
        '''
        return self._success_iter + self._fail_iter
    
    def _reset_run_stats(self):
        '''
        Resets the run statistics to 0
        '''
        self._success_iter = 0
        self._fail_iter = 0
    
    def _new_agent_executor(self) -> AgentExecutor:
        '''
        Resets the agent executor to it's initial state
        '''
        return self.executor_fn(**self.executor_kwargs)

    def _get_info(self) -> Dict[str, Any]:
        '''
        Get info to assign to the current benchmarking run
        '''
        # Get a new agent executor for benchmarking
        agent_executor = self._new_agent_executor()

        info = {}
        # TODO log LLM version

        # Log Tool(s)
        # TODO log tool source function
        info['tools'] = [
            {'name': tool.name, 'description': tool.description} for tool in agent_executor.tools
        ]

        # Log memory
        # If no memory is used
        if agent_executor.memory is None:
            info['memory'] = None
        # If multiple memory classes are used
        elif isinstance(agent_executor.memory, CombinedMemory):
            for i in range(len(agent_executor.memory.memories)):
                # TODO modify this to remove API keys
                info[f'memory_{i}'] = str(type(agent_executor.memory.memories[i]))
                info[f'buffer_{i}'] = str(agent_executor.memory.memories[i].buffer)
        # If a single memory class is used
        elif isinstance(agent_executor.memory, BaseMemory):
            info['memory'] = str(type(agent_executor.memory))
            info['buffer'] = str(agent_executor.memory.buffer)
        
        # Log questions
        info['initial_input'] = self.initial_input

        # Log Statistics
        info['total_iter'] = self.total_iter
        info['success_iter'] = self.success_iter
        info['fail_iter'] = self.fail_iter
        info['benchmark_score'] = (self.success_iter/self.total_iter)

        return info
    
    def _info_log(self, **dumps_kwargs):
        '''
        Log's final info JSON
        '''
        print('- Benchmarking Log:')
        print(json.dumps(self._get_info(), **dumps_kwargs))
    
    def _verbose_log(self, msg: str) -> None:
        '''
        Log's output in verbose mode to stdout (print)
        '''
        if self.verbose: print(msg)

    def _cast_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Casts the agent output to the desired format. Default is no casting
        '''
        return agent_output
    
    def _validate_agent_output(self, agent_output: Dict[str, Any]) -> None:
        '''
        Validates the agent output, raises error upon validation fail
        '''
        # Ensure the agent output is a dictionary
        if not isinstance(agent_output, dict):
            raise ValueError('Fatal error, Agent output must be a dictionary')
        # If a desired output key is not present in the agent output, the run will fail
        if any(key not in agent_output.keys() for key in self._desired_output.keys()):
            raise KeyError(
                f'''Fatal error, key not found\n 
                - desired output keys: {self._desired_output.keys()}
                - agent output keys: {agent_output.keys()}''')
        
    def _format_agent_output(self, agent_output: Any) -> str:
        '''
        Formats agent output as a string
        '''
        return f'- Agent Output: {agent_output}'.strip()

    def _compare_outputs(self, agent_output: Any, desired_output: List[Any], key: str) -> bool:
        '''
        Compares selected agent output to all desired outputs for that key.
        If it matches any of the desired outputs, it is considered a success
        '''
        result = any(self._compare_output(agent_output, output) for output in desired_output)

        if not result:
            self._verbose_log(f'- Fail on output key: {key}, did not match any of the desired outputs')
        if result:
            self._verbose_log(f'- Success on output key: {key}, matched a desired output')
        self._verbose_log(self._format_agent_output(agent_output))

        return result

    def _compare_output(self, agent_output: Any, desired_output: Any) -> bool:
        '''
        Compares selected agent output a desired output
        '''
        raise NotImplementedError

    def _validate_run(self, output: Dict[str, Any]) -> bool:
        '''
        Validates the current iteration of the benchmarking run
        '''
        # Validate that the agent output is in the correct format
        self._validate_agent_output(output)

        # Cast the agent output if needed
        output = self._cast_agent_output(output)
        
        # Validate all desired outputs
        validation_results = [self._compare_outputs(output[key], value, key) for key, value in self._desired_output.items()]

        # Return False if any of the results fail, otherwise True
        return not any(not result for result in validation_results)

    def _benchmark_loop(self) -> bool:
        '''
        Runs a single loop of the benchmarking agent
        '''
        # Reset the system
        self._reset_system()
        
        # Reset the agent
        agent_executor = self._new_agent_executor()
            
        # Run the agent
        output = agent_executor.invoke({'input': self.initial_input})

        # Validate the run
        return self._validate_run(output)
    
    def _reset_system(self) -> None:
        '''
        Resets the system to it's initial state if a reset function 
        is provided. 
        
        Confirms with the user that the reset is finished if validate
        reset is provided
        
        '''
        if isinstance(self.reset_system, Callable):
            self.reset_system()
        if self.validate_reset:
            input('Confirm when reset is finished')    

    def benchmark(
        self,
        iterations: int,
        initial_input: str | None = None,
        verbose: bool | None = None,
    ) -> None:
        '''
        Runs the benchmark 
        '''
        # Update initial question if needed
        if initial_input is not None:
            self.initial_input = initial_input

        if verbose is not None:
            self.verbose = verbose

        for i in self._tqdm(range(iterations)):
            # Run the loop
            try:
                loop_result = self._benchmark_loop()
            except Exception as e:
                print(f"Agentic Error: {e}")
                loop_result = False

            # Update success/fail
            if loop_result:
                self._success_iter += 1
            else:
                self._fail_iter += 1

            # Print if verbose
            if loop_result and self.verbose:
                print(f'- Iteration {i+1}: Success')
            if not loop_result and self.verbose:
                print(f'- Iteration {i+1}: Fail')

            if self.verbose:
                # Print output
                print(f'- Success: {self._success_iter}, Fail: {self._fail_iter}, Total: {self._success_iter+self._fail_iter}')
                print(f'- Benchmarking Score: {self._success_iter/(self._success_iter+self._fail_iter)}')
                print('--------------------')   

        self._info_log(indent=2)

    def async_bencmark(
        self,
        iterations: int,
        initial_input: str | None = None,
        verbose: bool | None = None,
    )-> None:
        '''
        Runs the benchmark asynchronously
        '''
        raise NotImplementedError