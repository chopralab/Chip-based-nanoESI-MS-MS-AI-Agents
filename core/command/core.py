from pydantic import (
    model_validator,
)
from typing import (
    Dict, 
    Any,
    List,
    Optional,
    Callable,
)
from sciborg_dev.core.parameter.base import Parameter
from sciborg_dev.core.command.base import BaseDriverCommand

class InteractiveParameterDriverCommand(BaseDriverCommand):
    '''
    Description
    -----------
    Driver command which allows for interactive parameter input to the command function
    at runtime. Interactive paramters can be included or excluded as needed.

    Example
    -------
    Using `include`
    ```
    self.parameter = [param_1, param_2, param_3]
    self.include = [param_1, param_3]
    ```
    Using `exclude`
    ```
    self.parameter = [param_1, param_2, param_3]
    self.exclude = [param_2]
    ```
    Will prompt
    ```text
    >>> Input value for parameter 'param_1': 1
    >>> Input value for parameter 'param_3': 3
    ```

    Attributes
    ----------
    ```
    uuid : str
    ```
    UUID of system associated with the driver command
    ```
    fn : Union[Callable, str]
    ```
    Function or function name accessed via import of python callable object to be used during
    `__call__` override of driver command object
    ```
    module : Optional[str] = None
    ```
    Name of the module to import if accessing function from that module
    ```
    package : Optional[str] = None
    ```
    Name of the package to be used during module import if needed
    ```
    has_return : Optional[bool] = False
    ```
    True if function has a return value and it should be accessed, False otherwise
    ```
    include : List[Parameter] | None = None
    ```
    A list of parameters to include in the interactive prompting (if exclude is not set)
    ```
    exclude : List[Paramter] | None = None
    ```
    A list of parameters to exclude from the interactive prompting (if include is not set)
    
    Methods
    -------
    ```
    def __call__(wf_globals: Dict[str, Any]=None, save_vars: Dict[str, str]=None, **kwargs) -> Dict
    ```
    Call the DriverCommand as a function (`self.fn` is called) with globals and kwargs as input.
    Output is saved off and/or returned as applicable
    '''

    # Public attributes
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_interactive_command(self) -> 'InteractiveParameterDriverCommand':
        # Set defaults for include and exclude
        if self.include is None: self.include = []
        if self.exclude is None: self.exclude = []

        # Make sure that include and exclude are not both set
        if len(self.include) > 0 and len(self.exclude) > 0:
            raise ValueError(f"Parameter include and exclude cannot both be set")
        
        if not(all(elem in [name for name in self.parameters.keys()] for elem in self.include)):
            raise ValueError(f"Include parameters {self.include} does not match parameters {self.parameters.keys()}")
        
        if not(all(elem in [name for name in self.parameters.keys()] for elem in self.exclude)):
            raise ValueError(f"Exclude parameters {self.exclude} does not match parameters {self.parameters.keys()}")

        return self

    def _assign_kwargs_interactive(self) -> None:
        '''
        Description
        -----------
        Assigns specified kwargs interactively based on `self.include` or `self.exclude`.
        Interactive assignment is done by prompting the user via python `input()` function
        and saving the output to the corresponding parameter if valid
        '''
        for name, param in self.parameters.items():
            # If there are parameters to include and that parameter is one, prompt
            if len(self.include) > 0 and any(elem == name for elem in self.include):
                param.value = input(f"Input value for parameter '{name}': ")
            # If there are parameter to exclude and that parameter is not one, prompt
            if len(self.exclude) > 0 and not any(elem == name for elem in self.include):
                param.value = input(f"Input value for parameter '{name}': ")

    def __call__(self, wf_globals: Dict[str, Any] = None, save_vars: Dict[str, str] = None, **kwargs) -> Dict:
        # Interactive kwarg assignment
        self._assign_kwargs_interactive()

        # Call the superclass method
        return super().__call__(wf_globals, save_vars, **kwargs)
    
class InteractiveResultDriverCommand(BaseDriverCommand):
    '''
    '''
    result_vars: List[str]
    has_return: bool = True
    fn: Optional[Callable] = None

    @model_validator(mode='after')
    def validate(self) -> Dict:
        self._init_private_attributes()
        return self

    def _init_private_attributes(self):
        def obtain_result(**kwargs):
            print(f"Run Parameters: {kwargs}", flush=True)
            if self.fn is not None:
                print(f"Helper Function {self.fn(**kwargs)}", flush=True)
            results = {}
            for elem in self.result_vars:
                results[elem] = float(input(f"Input value for result {elem}"))
            return results
        self._function = obtain_result