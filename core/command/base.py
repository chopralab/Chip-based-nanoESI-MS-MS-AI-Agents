from pydantic import (
    BaseModel,
    ConfigDict, 
    model_validator,
    field_validator,
    model_serializer,
    PrivateAttr,
    Field, 
)
from typing import (
    Dict, 
    Any,
    Type, 
    Optional,
    Union,
    Callable,
)
from types import ModuleType
from sciborg_dev.core.parameter.base import Parameter, ParameterModel, ValueType
import inspect
from importlib import import_module
from uuid import UUID
from pydoc import locate

class BaseCommand(BaseModel):
    """
    Description
    -----------
    Base model for all commands. 
    This class has a dictionary of parameters associated with the command.
    It is reccomended that one of the three other base commands be extended
    for additional functionality instead of this command.

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the command
    ```
    parameters : Dict[str, Parameter] | None
    ```
    A dictionary of parameters associated with the command

    Methods
    -------
    ```
    def values(from_vars: bool=False) -> Dict[str, Any]
    ```
    Gets the parameter dictionary as a dictionary which replaces the Parameter
    object with the value of the Parameter object.

    If from_vars is set to True, it will replace any parameter with from_vars
    equal to True to "_var.{var_name}"
    ```
    def __setitem__(name: str, value: Any) -> None
    ```
    Sets the specified parameter to the specificed value
    ```
    def __getitem__(name: str) -> Any
    ```
    Gets the value of the specified parameter
    """

    # Command model config
    model_config = ConfigDict(validate_assignment=True)

    # Command Model attributes
    name: str
    microservice: str
    uuid: UUID
    desc: str = ""
    parameters: Dict[str, ParameterModel] | None = {}
    
    def __setitem__(self, key:str, value: ParameterModel) -> None:
        """
        Descirption
        -----------
        Sets the specified parameter to the specificed value. A validation error
        will occur if the new value is invalid. A key error will occur if the
        key is not present.

        Parameters
        ----------
        ```
        name: str
        ```
        The name of the parameter to update
        ```
        value: Any
        ```
        The value of the parameter to update to
        """
        self.parameters.__setitem__(key, value)

    def __getitem__(self, key:str) -> ParameterModel:
        """
        Description
        -----------
        Gets the value of the specified parameter. Raises a 
        key error if the key does not exist

        Parameters
        ----------
        ```
        name: str
        ```
        The name of the parameter to access

        Return
        ------
        ```
        value : Any
        ```
        The value of the parameter that was accessed
        """
        return self.parameters.__getitem__(key)
    
    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()
    
    def items(self):
        return self.parameters.items()
        
class BaseRunCommand(BaseCommand):
    '''
    Description
    -----------
    Base model for a run command which specifies a command to be run on a specific endpoint

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the command
    ```
    parameters : Dict[str, Parameter] | None
    ```
    A dictionary of parameters associated with the command
    ```
    uuid : str
    ```
    The UUID of the endpoint
    ```
    save_vars : Dict[str, str]
    ``` 
    Specify which keys of the command output to save off to specific variables
    of a global dictionary after command execution
    '''

    # Run Command attributes
    desc: str = Field("", exclude=True)
    save_vars: Dict[str, str] = {}
    parameters: Dict[str, Parameter] = {}

    def values(self):
        return self.parameters.values()
    
    def to_kwargs(self) -> Dict[str, Any]:
        return {key:parameter.value for key,parameter in self.parameters.items() if not parameter.from_var}
    
    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()
    
    def items(self):
        return self.parameters.items()
    
class BaseInfoCommand(BaseCommand):
    '''
    Description
    -----------
    Base model for a library command which acts as a standardized command,
    abstracting out all information on command execution.

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the command.
    ```
    parameters : Dict[str, Parameter] | None
    ```
    A dictionary of parameters associated with the command.
    ```
    has_return : bool
    ```
    True if the command returns information, False otherwise.
    ```
    return_signature : Dict[str, Type] | None
    ```
    A signature of the expected return, can be a dictionary or None.

    Methods
    -------
    ```
    def to_run_command(self, uuid: str, save_vars: Union[Dict[str,str], None] = None) -> BaseRunCommand
    ```
    Dynamically builds a run command to be run on a specific endpoint 
    based on the current LibraryCommand.
    ```
    def to_message(self, uuid: str, save_vars: Dict[str,str]=None) -> str
    ```
    Dynamically builds a message protocol of a run command for a 
    specific endpoint based on the current Library Command.
    '''

    # Library command attributes
    has_return: bool = True
    return_signature: Dict[str, str] | None = None

    @model_validator(mode='after')
    def validate_info_command(self):
        if not self.has_return:
            self.return_signature = None

    def to_run_command(
            self,
            var_names: Dict[str, str] | None = None,
            save_vars: Optional[Dict[str,str]] = None,
            **kwargs
        ) -> BaseRunCommand:
        '''
        Description
        -----------
        Dynamically builds a run command to be run on a specific endpoint 
        based on the current LibraryCommand.

        Parameters
        ----------
        ```
        uuid : str
        ```
        The UUID of the run endpoint.
        ```
        save_vars : Optional[Dict[str, str]] = None
        ```
        A dictionary of variables to save off command output to.

        Return
        ------
        ```
        run_command : BaseRunCommand
        ```
        A new RunCommand object which corresponds to the current LibraryCommand.
        '''

        # Set defaults
        if var_names is None: var_names = {}
        if save_vars is None: save_vars = {}

        # Validate keys in save_vars if return_signature is specified
        if self.return_signature:
            for key in save_vars.keys(): 
                if key not in self.return_signature.keys(): 
                    raise KeyError(f"Key: '{key}' is not a valid return key, expected one of: {self.return_signature.keys()}")

        if any(key not in self.parameters.keys() for key in var_names.keys()):
            raise KeyError(
                f"""All variable name keys must correspond to a parameter:
                variable name keys: {var_names.keys()}
                parameters keys: {self.parameters.keys()}"""
            )
        
        if any(key not in self.parameters.keys() for key in kwargs.keys()):
            raise KeyError(
                f"""All additional argument keys must correspond to a parameter:
                kwargs keys: {kwargs.keys()}
                parameters keys: {self.parameters.keys()}"""
            )

        # Build parameters from parameter model
        parameters: Dict[str, Parameter] = {}
        for key, value in self.parameters.items():
            Value = value.to_param()
            parameters[key] = Value()

            # Assign variable templates
            if key in var_names.keys():
                parameters[key].from_var = True
                parameters[key].var_name = var_names[key]

            if key in kwargs.keys():
                parameters[key].value = kwargs[key]

        return BaseRunCommand(
            **{
            "uuid": self.uuid, 
            "name": self.name,
            'microservice': self.microservice,
            "parameters": parameters,
            "save_vars" : save_vars.copy()
            })
    
class BaseDriverCommand(BaseCommand):
    '''
    Note
    ----
    NOTE: We may want to redo the base class to be a skeleton class 
    and then have this class be a function call driver command. The
    reason for this is that this base driver command structure assumes that
    we are running fn when overriding `__call__` but we may overide call with
    the function built in directly

    Description
    -----------
    Driver command class for execution of an arbitrary python function with arguments
    provided at runtime. Function can be passed directly or imported from a module
    Arguments are checked against command parameters to ensure 
    validity based on defined structure.

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
    _module (Private): ModuleType = None
    ```
    The module which is imported if `module` attribute is set
    ```
    _function (Private): Callable = None
    ```
    The function which will be used during `__call__` override

    Methods
    -------
    ```
    def __call__(wf_globals: Dict[str, Any]=None, save_vars: Dict[str, str]=None, **kwargs)
    ```
    Call the DriverCommand as a function (`self.fn` is called) with globals and kwargs as input.
    Output is saved off and/or returned as applicable
    '''

    # DriverCommand public attributes
    fn: Callable | str
    module: str | None = None
    package: str | None = None
    has_return: bool = True
    return_signature: Dict[str, str] | None = None

    # DriverCommand private attributes
    _module: ModuleType | None = PrivateAttr(default=None)
    _function: Callable | None = PrivateAttr(default=None)
    _parameters: Dict[str, Parameter] | None = PrivateAttr(default=None)

    @model_serializer(when_used='json')
    def serialize_base_driver_command(self) -> Dict:
        json_dumps = self.model_dump()
        if isinstance(json_dumps['fn'], Callable):
            # If we have a Callabe, we get the name of the function
            json_dumps['fn'] = json_dumps['fn'].__name__
        return json_dumps
    
    def _validate_module(self) -> Union[Callable, None]:
        '''
        '''
        # If we are importing from a module, the module must be defined
        if isinstance(self.fn, str) and self.module is None:
            # TODO build in compile from source code if possible
            raise TypeError("Module cannot be None with function type str")
        
        # We are importing from a module
        if self.module is not None and isinstance(self.fn, str):
            # Make sure the module can be imported
            self._module = import_module(self.module, self.package)
            if self._module is None:
                raise ModuleNotFoundError(f"Import on module: {self.module} with package: {self.package} failed")
            
            # If we are importing from a module, the function needs to be a string
            # if not isinstance(self.fn, str):
                # raise TypeError(f"Only function names (str) may be passed in with module import, received {type(self.fn)}")

            # Assign the function to the private attribute
            self._function = self._module.__getattribute__(self.fn)

    def _validate_function(self) -> None:
        '''
        '''
        # Assign if we did not import from a module
        if self._function is None: self._function = self.fn

        # Make sure the function is a Callable type
        if not isinstance(self._function, Callable):
            raise TypeError("Function type must be Callable with module set to None")
        
        # Grab module from function if possible
        if self._module is None:
            self._module = inspect.getmodule(self._function)
            if self._module is not None:
                self.module = self._module.__name__
                self.package = self._module.__package__

    def _validate_parameters(self) -> None:
        # Make sure the command arguments match with parameters
        for arg in list(inspect.signature(self._function).parameters.keys()):
            if arg not in self.parameters.keys():
                raise KeyError(f"Argument {arg} is not a command parameter, must be one of {self.parameters.keys()}")

    def _validate_return_signature(self) -> None:
        # Removing the dictionary check for now as its not the best way to do this
        # if self.has_return and not isinstance(self.return_signature, dict):
        #     raise ValueError("If has return is specified, return signature must be a dictionary")
        # if self.has_return and not self.return_signature:
        #     raise ValueError("If has return is True, the return signature must not be an empty dictionary.")
        if not self.has_return:
            self.return_signature = None

    def _init_private_attributes():
        pass

    @model_validator(mode='after')
    def validate(self) -> 'BaseDriverCommand':
        self._init_private_attributes()
        # Define/validate fn from module if possible
        self._validate_module()

        # Define/validate fn
        self._validate_function()

        # Validate function parameters from signature
        self._validate_parameters()

        # Validate the return signature
        self._validate_return_signature()

        return self

    def _validate_kwargs(self, **kwargs: Dict[str, Any]) -> None:
        '''
        Description
        -----------
        Checks kwargs for invalid arguments based on DriverCommand parameters.
        Raises an KeyError if any invalid parameters are found.

        Parameters
        ----------
        ```
        **kwargs : Dict[str, Any]
        ```
        Arguments to be passed into the command function at runtime.
        '''
        for key in kwargs.keys():
            if key not in self._parameters.keys():
                raise KeyError(f"Key: '{key}' not found, must be one of {self._parameters.keys()}")

    def _init_private_attributes(self):
        self._set_parameters()

    def _set_parameters(self):
        if self._parameters is None:
            parameters = {}
            for key, parameter in self.parameters.items():
                parameters[key] = parameter.to_param()()
            self._parameters = parameters

    def _update_parameters(self, wf_globals: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        '''
        Description
        -----------
        Updates command parameters based on kwargs and workflow globals. 
        Reverts upon invalid parameter assignment and raises an error.

        Parameters
        ----------
        ```
        wf_globals : Dict[str, Any]
        ```
        Global varaibles of a workflow which can be passed into the command at runtime
        ```
        **kwargs : Dict[str, Any]
        ```
        Arguments to be passed into the command function at runtime
        '''

        # Save off previous values incase we need to revert after an exception is raised
        prev_args = dict(zip(self._parameters.keys(), list(map(lambda o: o.value, list(self._parameters.values())))))

        # Update parameters, revert upon invalid assignment
        for key, value in kwargs.items():
            try: self._parameters[key].value = value
            except Exception as e:
                # Revert all parameters to previous values and raise error
                for prev_key, prev_value in prev_args.items():
                    self._parameters[prev_key].value = prev_value 
                raise e
        
        # If any parameters are from workflow globals, update here
        for key, param in self._parameters.items():
            # If the parameter is from a varaible and that varaible exists in global
            if param.from_var and param.var_name in wf_globals.keys():
                # Assign that parameter to the global if possible
                try: self._parameters[key].value = wf_globals[param.var_name]
                except Exception as e:
                    # Revert all parameters to previous values and raise error
                    for prev_key, prev_value in prev_args.items():
                        self._parameters[prev_key].value = prev_value 
                    raise e

    def _save_results_to_globals(self, result: Dict[str, Any], wf_globals: Dict[str, Any], save_vars: Dict[str, Any]) -> None:
        '''
        Description
        -----------
        Saves the results of a command to the workflow globals based on varaible names in `save_vars`.

        Parameters
        ----------
        ```
        result : Dict[str, Any]
        ```
        Results of the current command after execution
        ```
        wf_globals : Dict[str, Any]
        ```
        Global varaibles of the workflow which will have values saved to them
        ```
        save_vars : Dict[str, Any]
        ```
        Names of varaibles in results to save to the workflow globals.

        Saving is done in the following manner:
        ```
        save_vars.keys()   # Name of varaible in results to save
        save_vars.values() # Name of varaible in wf_globals to save to
        ```
        '''
        for result_var, global_var in save_vars.items():
            wf_globals[global_var] = result[result_var]


    def __call__(self, wf_globals: Dict[str, Any]=None, save_vars: Dict[str,str]=None, **kwargs) -> Dict:
        '''
        Description
        -----------
        Calls the driver command function with provided global dictionary and arguments.
        Results are saved in the global dictionary as specified and returned (if applicable)

        Parameters
        ----------
        ```
        wf_globals : Dict[str, Any] = None
        ```
        Dynamic arguments to call the driver command function with, must match with `self.parameter`.
        Specified results are saved to this dictionary after command execution.
        ```
        save_vars : Dict[str,str] = None
        ```
        Dictionary of varaibles to save off from command output to wf_globals
        ```
        **kwargs: Dict[str, Any]
        ```
        Static arguments to call the driver command function with ,must match with `self.parameters`

        Return
        ------
        ```
        results : Dict
        ```
        The results of the command execution (`self.fn` output). Must be a dictionary.
        '''
        
        # Make sure that the function is implemented prior to call
        if self._function is None:
            raise NotImplementedError("Function is not implemented, check class initilization")

        # If dictionaries are not provided, set to empty dictionaries
        if save_vars is None: save_vars={}
        if wf_globals is None: wf_globals = {}

        # Set or reset parameters to defaults prior to function call
        # self._set_parameters()

        # Check for invalid arguments
        self._validate_kwargs(**kwargs)

        # Update parameters based on kwargs and wf_globals
        self._update_parameters(wf_globals, **kwargs)

        # Extract function arg values from parameter objects
        args = dict(zip(self._parameters.keys(), list(map(lambda o: o.value, self._parameters.values()))))

        # Call function and return result if applicable
        result = self._function(**args)
        # if self.has_return and isinstance(result, dict):
        #     # Save off varibles to workflow globals (if applicable) and return result
        #     self._save_results_to_globals(result, wf_globals, save_vars)
        #     return result
        if self.has_return:
            # Save off varibles to workflow globals (if applicable) and return result
            self._save_results_to_globals(result, wf_globals, save_vars)
            return result
        
    def to_info_command(self) -> BaseInfoCommand:
        return BaseInfoCommand(
            name=self.name,
            uuid=self.uuid,
            desc=self.desc,
            microservice=self.microservice,
            parameters=self.parameters,
            has_return=self.has_return,
            return_signature=self.return_signature,
        )

    def set_param_values_from_run_command(self, command: BaseRunCommand) -> None:
        for param_name, parameter in command.items():
            if parameter.from_var is False:
                self._parameters[param_name].value = parameter.value
            else: 
                self._parameters[param_name].from_var = parameter.from_var
                self.parameters[param_name].from_var = parameter.from_var
                self._parameters[param_name].var_name = parameter.var_name
                self.parameters[param_name].var_name = parameter.var_name
    
    def keys(self):
        return self._parameters.keys()
    
    def values(self):
        return self._parameters.values()
    
    def items(self):
        return self._parameters.items()
    
    def model_keys(self):
        return self.parameters.keys()
    
    def model_values(self):
        return self.parameters.values()
    
    def model_items(self):
        return self.parameters.items()
    
    def get_param_model(self, key: str) -> ParameterModel:
        return self.parameters.__getitem__(key)
    
    def set_param_model(self, key: str, value: ParameterModel) -> None:
        return self.parameters.__setitem__(key, value)

    def __getitem__(self, key: str) -> ValueType:
        return self._parameters.__getitem__(key).value
    
    def __setitem__(self, key: str, value: ValueType) -> None:
        self._parameters.__getitem__(key).value = value

    def set_var_name(self, param_name: str, var_name: str) -> None:
        self._parameters.__getitem__(param_name).set_var_name(var_name)