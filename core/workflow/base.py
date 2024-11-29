from pydantic import BaseModel, InstanceOf
from typing import (
    Dict, 
    Any,
    List, 
    Union,
    Optional,
    SupportsIndex,
    Iterator,
)
from sciborg_dev.core.command.base import (
    BaseCommand,
    BaseInfoCommand, 
    BaseRunCommand, 
    BaseDriverCommand,
)

class BaseWorkflow(BaseModel):
    '''
    Description
    -----------
    Base model for all workflows

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the workflow
    ```
    command : List[BaseCommand]
    ```
    Generic command list for the workflow
    '''
    name: str
    commands: List[BaseCommand]

    def append(self, command: BaseCommand) -> None:
        self.commands.append(command)

    def __len__(self) -> int:
        return self.commands.__len__()

    def __getitem__(self, i: SupportsIndex | slice) -> BaseCommand | List[BaseCommand]:
        return self.commands.__getitem__(i)
    
    def __iter__(self) -> Iterator[BaseCommand]:
        return self.commands.__iter__()

class BaseRunWorkflow(BaseWorkflow):
    '''
    Description
    -----------
    Base model for all workflows which contain RunCommands

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the workflow
    ```
    command : List[BaseRunCommand]
    ```
    A list of RunCommands (in order) which defines the workflow steps

    Methods
    -------
    ```
    def m_dict() -> List[Dict]:
    ```
    Returns a list of all run commands represented as a dictionary formatted for messaging.
    ```
    def m_json() -> List[str]:
    ```
    Returns a list of JSON formatted message protocols for the run commands
    '''
    commands: List[BaseRunCommand]

    def append(self, command: BaseRunCommand) -> None:
        return super().append(command)
    
    def __getitem__(self, i: SupportsIndex | slice) -> BaseRunCommand | List[BaseRunCommand]:
        return super().__getitem__(i)
    
    def __iter__(self) -> Iterator[BaseRunCommand]:
        return super().__iter__()

class BaseInfoWorkflow(BaseWorkflow):
    '''
    Description
    -----------
    Base model for all workflows which contain LibraryCommands

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the workflow
    ```
    commands : List[BaseLibraryCommand]
    ```
    A list of LibraryCommands (in order) which define the workflow steps

    Methods
    -------
    ```
    def to_run_workflow(self, uuids: List[str], save_var_list: List[Dict[str, str]] = None) -> BaseRunWorkflow
    ```
    Converts a library workflow to a run workflow for a specific set of UUIDs.
    This run workflow will be sent as messages to a remote machine for execution.
    ```
    def to_message_workflow(self, uuids: List[str], save_var_list: List[Dict[str, str]] = None) -> List[str]
    ```
    Converts a library workflow to a list of messages for a specific set of UUIDs
    '''
    commands: List[InstanceOf[BaseInfoCommand]]

    def to_run_workflow(
        self,
        var_name_list: List[Dict[str, str]] = None,
        save_var_list: List[Dict[str, str]] = None
    ) -> BaseRunWorkflow:
        '''
        Description
        -----------
        Converts a library workflow to a run workflow for a specific set of UUIDs.

        This run workflow will be sent as messages to a remote machine for execution.

        Parameters
        ----------
        ```
        uuid : List[str]
        ```
        The uuids to be applied to self.commands.
        Must be the same length and order as the self.commands.
        ```
        save_var_list : List[Dict[str, str]]
        ```
        A list of dictionaries which contain pairings on which varaibles will be saved off to which globals upon command completion
            
        Example:

        ```
        self.commands = [command_1, command_2]
        save_var_list = [{"product_smiles", "smiles"}, {"score": "sa_score"}]
        ```

        Assuming `command_1` returns the following dictionary:

        ```
        {"smiles": "CC=OOCC"}
        ```

        And `command_2` returns the following dictionary:
        ```
        {"sa_score": 2.5}
        ```

        The driver workflow should have the following global dictionary after execution
        ```
        {"product_smiles": "CC=OOCC","score": 2.5}
        ```

        Returns
        -------
        ```
        run_workflow : BaseRunWorkflow
        ```
        A RunWorkflow for a specific set of UUIDs based on the current LibraryWorkflow.
        '''
        if save_var_list is None: save_var_list = len(self.commands)*[None]
        if var_name_list is None: var_name_list = len(self.commands)*[None]
        
        if len(self.commands) != len(save_var_list):
             raise ValueError(f"Command and Var list must be of the same length, command list length: {len(self.commands)} != UUID list length: {len(save_var_list)}")
        
        return BaseRunWorkflow(
            name=f"{self.name}_run",
            commands=[lib_command.to_run_command(uuid, var_names, save_vars) for lib_command, uuid, var_names, save_vars in list(zip(self.commands, var_name_list, save_var_list))]
        )
    
    def append(self, command: BaseInfoCommand) -> None:
        return super().append(command)
    
    def __getitem__(self, i: SupportsIndex | slice) -> BaseInfoCommand | List[BaseInfoCommand]:
        return super().__getitem__(i)
    
    def __iter__(self) -> Iterator[BaseInfoCommand]:
        return super().__iter__()

class BaseDriverWorkflow(BaseWorkflow):
    '''
    Description
    -----------
    Base model for all workflow which contain driver commands

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the workflow
    ```
    commands : List[BaseDriverCommand]
    ```
    A list of DriverCommands (in order) which define the workflow steps
    ```
    wf_globals: Dict[str, Any]
    ```
    A dictionary of global varaibles that are shared between all workflow command during execution.
    Commands can read in from and write out to this list.

    Methods
    -------
    ```
    def clear_wf_globals()
    ```
    Resets `self.wf_globals` to an empty dictionary.
    ```
    def exec(self, list_kwargs: List[Union[Dict[str, Any], None]], list_save_vars: List[Union[Dict[str,str], None]]) -> List[Dict]
    ```
    Executes the DriverWorkflow (in order) with each command provided with workflow gloabs and its own set of kwargs.
    '''
    commands: List[BaseDriverCommand]
    wf_globals: Dict[str, Any] = {}

    def clear_wf_globals(self) -> None:
        '''
        Description
        -----------
        Resets `self.wf_globals` to an empty dictionary 
        '''
        self.wf_globals = {}

    def exec(self, list_kwargs: List[Optional[Dict[str, Any]]] = None, list_save_vars: List[Optional[Dict[str,str]]] = None) -> List[Dict]:
        '''
        Description
        -----------
        Executes the DriverWorkflow (in order) with each command provided with workflow gloabs and its own set of kwargs.

        Parameters
        ----------
        ```
        list_kwargs : List[Dict[str, Any] | None]
        ```
        A list of kwargs to be provided to the corresponding DriverCommand in the workflow
        ```
        # Assuming
        self.commands = [command_1, command_2]
        list_kwargs = [{"arg1": 1, "arg2": 2}, {"arg3": 3, "arg4": 4}]
        # Will call during execution
        command_1(**{"arg1": 1, "arg2": 2})
        command_2(**{"arg3": 3, "arg4": 4})
        ```
        ```
        list_save_vars : List[Dict[str, str] | None]
        ```
        A list of dictionaries which contain pairings on which varaibles will be saved off to which globals upon command completion
            
        Example:

        ```
        self.commands = [command_1, command_2]
        save_var_list = [{"product_smiles", "smiles"}, {"score": "sa_score"}]
        ```

        Assuming `command_1` returns the following dictionary:

        ```
        {"smiles": "CC=OOCC"}
        ```

        And `command_2` returns the following dictionary:
        ```
        {"sa_score": 2.5}
        ```

        The driver workflow should have the following global dictionary after execution
        ```
        {"product_smiles": "CC=OOCC","score": 2.5}
        ```
        Returns
        -------
        ```
        result_log : List[Dict[str, Any]]
        ```
        A list of dictionarys resulting from DriverCommand execution of the workflow
        '''

        # Assign defaults if needed
        if list_kwargs is None: list_kwargs = len(self.commands)*[{}]
        if list_save_vars is None: list_save_vars = len(self.commands)*[{}]

        # Replace None with empty dictionary
        for i in range(len(list_kwargs)):
            if list_kwargs[i] is None:
                list_kwargs[i] = {}
        for i in range(len(list_save_vars)):
            if list_save_vars[i] is None:
                list_save_vars[i] = {}

        # Check to ensure all lists are equal
        if len(self.commands) != len(list_kwargs):
            raise ValueError(f"Command and argument lists must be of the same length, command list length: {len(self.commands)} != kwarg list length: {len(list_kwargs)}")
        if len(self.commands) != len(list_save_vars):
            raise ValueError(f"Command and save var lists must be of the same length, command list length: {len(self.commands)} != kwarg list length: {len(list_save_vars)}")
        
        # Pair off commands with corresponding kwargs
        exec_commands = list(zip(self.commands, list_kwargs, list_save_vars))

        result_log = []
        for command, kwargs, save_vars in exec_commands:
            # Run the command with provided globals and kwargs
            result = command(wf_globals=self.wf_globals, save_vars=save_vars, **kwargs)
            result_log.append(result)
        
        return result_log
    
    def to_info_workflow(self) -> BaseInfoWorkflow:
        '''
        Converts the driver workflow in its current state into a information workflow
        '''
        return BaseInfoWorkflow(
            name=self.name,
            commands=[driver_command.to_info_command() for driver_command in self.commands]
        )

    def append(self, command: BaseDriverCommand) -> None:
        return super().append(command)
    
    def __getitem__(self, i: SupportsIndex | slice) -> BaseDriverCommand | List[BaseDriverCommand]:
        return super().__getitem__(i)
    
    def __iter__(self) -> Iterator[BaseDriverCommand]:
        return super().__iter__()