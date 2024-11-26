from pydantic import BaseModel, ConfigDict, Field, validate_call, field_validator, InstanceOf
from typing import List, Any, Dict

from sciborg.core.library.base import BaseDriverCommandLibrary
from sciborg.core.command.base import BaseRunCommand, BaseDriverCommand
from sciborg.core.workflow.base import BaseDriverWorkflow, BaseRunWorkflow

class BaseCommandInterpreter(BaseModel):
    '''
    Base class for a LINQX command interpreter.

    This interpreter is designed to run driver side and converts message based (run) protocols
    to executable commands. The interpreter has a library of driver (operational) commands that 
    it can select from.

    ### Attributes
    ```
    driver_library: BaseDriverCommandLibrary # Internal command library for the interpreter
    ```
    ### Methods
    ```
    def interpret_workflow(
        self,
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow,
        name: str = "undefined",
    ) -> BaseDriverWorkflow: # Interprets the text-based run workflow instructions to a executable driver workflow

    def interpret_and_run_workflow(
        self,
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow,
        name: str = "undefined",
    ) -> List[Dict]: # Interprets and execute the text-based run workflow
    ```

    '''
    model_config = ConfigDict(validate_assignment=True)

    '''
    Either I am missing something or there is an issue with Pydantic field validation.

    We have switched to using `InstanceOf` instead of just the driver command model as
    valid driver command objects were causing the attribute to set to `None` without 
    throwing an error or setting to the default value. 
    ''' 
    driver_library: InstanceOf[BaseDriverCommandLibrary] = Field(default=BaseDriverCommandLibrary(name='default'), description='Internal command library for the interpreter')

    @staticmethod
    def _cast_workflow(
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow,
        name:str = 'undefined'
    ) -> BaseRunWorkflow:
        '''
        Cast run command or a list of run commands to a run workflow if possible
        '''
        if isinstance(run_workflow, BaseRunWorkflow):
            return run_workflow
        elif isinstance(run_workflow, BaseRunCommand):
            return BaseRunWorkflow(name=name, commands=[BaseRunCommand])
        elif isinstance(run_workflow, list):
            return BaseRunWorkflow(name=name, commands=run_workflow)
        else:
            raise TypeError(f'Cannot runnable workflow from type: {type(run_workflow)}')

    def _interpret_command(self, run_command: BaseRunCommand) -> BaseDriverCommand:
        '''
        Interprets the provided RunCommand to a DriverCommand
        '''
        
        if str(run_command.uuid) not in self.driver_library.uuid_keys():
            raise KeyError(f"Microservice UUID {run_command.uuid} not found in command library")

        if run_command.name not in self.driver_library[run_command.uuid].keys():
            raise KeyError(f"Run command {run_command.name} not found in command library")
        
        return self.driver_library[run_command.uuid][run_command.name]

    @validate_call
    def interpret_workflow(
        self,
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow,
        name: str = "undefined",
    ) -> BaseDriverWorkflow:
        '''
        Interprets a run workflow or a list of run commands to a driver workflow.

        ### Parameters
        ```
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow, # The workflow to interpret
        name: str = "undefined", # The name of the workflow
        ```
        ### Returns
        ```
        return BaseDriverWorkflow # The interpreted driver workflow
        ```
        '''

        # Cast to run workflow if needed
        run_workflow = BaseCommandInterpreter._cast_workflow(run_workflow, name)

        # Create empty driver workflow
        driver_workflow = BaseDriverWorkflow(name=run_workflow.name, commands=[])
        
        # Add copy of driver commands to the new workflow
        for run_command in run_workflow.commands:
            # Get copy of the relevant driver command
            driver_command = self.driver_library[run_command.microservice][run_command.name]
            driver_command = driver_command.model_copy()

            # Assign additional parameter attributes to driver command
            driver_command.set_param_values_from_run_command(run_command)

            # Append the updated command to the workflow
            driver_workflow.append(driver_command)

        return driver_workflow
    
    @validate_call
    def interpret_and_run_workflow(
        self,
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow,
        name: str = "undefined",
    ) -> List[Dict]:
        '''
        Interprets a run workflow or a list of run commands to a driver workflow and then executes the driver workflow.

        ### Parameters
        ```
        run_workflow: BaseRunCommand | List[BaseRunCommand] | BaseRunWorkflow, # The workflow to interpret
        name: str = "undefined", # The name of the workflow
        ```
        ### Returns
        ```
        return BaseDriverWorkflow # The interpreted driver workflow
        ```
        '''
        # Cast to run workflow if needed
        run_workflow = BaseCommandInterpreter._cast_workflow(run_workflow, name)

        # Interprets the workflow
        driver_workflow = self.interpret_workflow(run_workflow, name)
        list_kwargs = []
        list_save_vars = []
        for run_command in run_workflow:
            list_kwargs.append(run_command.to_kwargs())
            list_save_vars.append(run_command.save_vars)

        # Executes the workflow and returns the log
        return driver_workflow.exec(list_kwargs, list_save_vars)