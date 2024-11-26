from pydantic import (
    BaseModel,
    ConfigDict, 
    PositiveInt, 
    PrivateAttr, 
    create_model,
    model_validator,
    field_validator,
    model_serializer
)
from typing import (
    List, 
    Dict, 
    Union, 
    Any, 
    Literal, 
    Optional, 
    Type,
    Callable,
    Tuple,
)

ValueType = int | str | float | bool | List[int | str | float | bool]

class Parameter(BaseModel):
    '''
    Description
    -----------
    This is a shell parameter class that extends BaseModel

    It is used to provide a uniform superclass when dynamically creating 
    parameters using the create_model() method

    Attributes
    ----------
    ```
    value : Union[int, str, float, List[Union[int, str, float]]]
    ```
    The value of the parameter
    ```
    desc : Optional[str] = ""
    ```
    A description of the parameter
    ```
    from_var: Optional[bool] = False
    ```
    A flag to denote wheter the parameter should be read from a workflow global
    varaible at runtime during workflow execution. See `commands.py` and `workflow.py`
    for more details.
    ```
    var_name : Optional[str] = ""
    ```
    The name of the varaible which will have its value assigned to the parameter
    upon workflow execution.
    ```
    upper_limit : Optional[Union[int, str, float]] = None
    ```
    The upper limit of the parameter, must match the data type.
    Cannot be less than the lower limit.
    ```
    lower_limit : Optional[Union[int, str, float]] = None
    ```
    The lower limit of the parameter, must match the data type.
    Cannot be greater than the upper limit.
    '''

    # Model config
    model_config = ConfigDict(validate_assignment=True)

    # Parameter attributes
    value: ValueType | None = None
    desc: str = ""
    from_var: bool = False
    var_name: str = ""

    @model_serializer(when_used='json')
    def serialize_parameter(self) -> Dict[str, Any]:
        '''
        Decorators
        ----------
        ```
        @model_serializer(when_used='json')
        ```
        Description
        -----------
        Override standard JSON serializer to return only the Parameter
        value

        Returns
        -------
        ```
        return self.value
        ```
        The value of the parameter
        '''
        return {
            'value': self.value,
            'from_var': self.from_var,
            'var_name': self.var_name,
        }
    
    def set_var_name(self, var_name: str) -> None:
        self.from_var = True
        self.var_name = var_name

class ParameterModel(BaseModel):
    '''
    Description
    -----------
    This class is used to define a custom model for a parameter to be use in LINQX
    infrastructure components. After initilizing the model, use the to_param()
    method to create a class which represents the parameter that was defined.
    From there, individual objects of that parameter can be created and reference
    in other LINQX infrastrucutre components.

    For example, to create a parameter called voltage which ranges between 60.0v and 240.0v
    and has a default value of 120.0v :

    ```python
    # Define the voltage model
    voltage_model = ParameterModel(
        name="Voltage",
        data_type="float",
        upper_limit=240.0,
        lower_limit=60.0,
        default=120.0,
        description="Voltage in volts"
    )

    # Build the Voltage class
    Voltage = voltage_model.to_param() # Subclass of Parameter
    
    # Build voltage objects
    v1 = Voltage() # -> has value of 120.0
    v2 = Voltage(value=220.0) # -> has value of 220.0
    v3 = Voltage(value=80.0) # -> has value of 80.0

    # It will prevent invalid objects from being build
    v_invalid = Voltage(value=300.0) # This will raise an error
    
    # Operators are supported
    v1 = v1 + 40
    v1.value # -> should be 160.0
    v1 == 160.0 # -> should be True
    ```

    It is also possible to define parameters which can have dynamically assigned 
    values during workflow execution. This is done by defining the parameter to 
    read from a specific varaible. See the example below:

    ```python
    # Assign v1 to read from 'voltage_1' dynamically
    v1.from_var = True
    v1.var_name = "voltage_1"

    # Assign v2 to read from 'voltage_2' dynamically
    v2.from_var = True
    v2.var_name = "voltage_2"
    ```

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the parameter model
    ```
    data_type : Literal["str", "int", "float"]
    ```
    The data type of the parameter model. Supports only primitives.
    Must be a string. 
    ```
    precision : Optional[Union[Literal[-1], PositiveInt]] = -1
    ```
    Precision to round floats to if applicable (-1 is infinite precision)
    ```
    upper_limit : Optional[Union[int, str, float]] = None
    ```
    The upper limit of the parameter, must match the data type.
    Cannot be less than the lower limit.
    ```
    lower_limit : Optional[Union[int, str, float]] = None
    ```
    The lower limit of the parameter, must match the data type.
    Cannot be greater than the upper limit.
    ```
    allowed_value : Optional[List[Union[int, str, float]]] = []
    ```
    A list of allowed values to restrict the parameter.
    Must match the data type of the parameter
    ```
    is_optional : Optional[bool] = False
    ```
    A flag to denote whether the parameter is required or optional.
    ```
    is_list : Optional[bool] = False
    ``` 
    A flag to denote whether the parameter is a value or a list.
    ```
    default : Optional[Union[List[Union[int, str,float]], int, str, float]] = None
    ```
    A default value for the parameter, must match type and obey limits.
    ```
    from_var : Optional[bool] = False
    ```
    A flag to denote wheter the parameter should be read from a workflow global
    varaible at runtime during workflow execution. See `commands.py` and `workflow.py`
    for more details.
    ```
    var_name : Optional[str] = ""
    ```
    The name of the varaible which will have its value assigned to the parameter
    upon workflow execution.
    ```
    desc : Optional[str] = ""
    ```
    A description of the parameter

    Methods
    -------
    ```
    def to_param() -> type[Parameter]
    ```
    Builds a `Parameter` subclass based on the `ParameterModels`
    specifications of the parameter
    '''
    # Parameter model attributes
    name: str
    data_type: Literal["str", "int", "float", "bool"]
    precision: Literal[-1] | PositiveInt = -1
    upper_limit: int | str | float | None = None
    lower_limit: int | str | float | None = None
    allowed_values: List[int | str | float] = []
    is_optional: bool = False
    is_list: bool = False
    default: ValueType | None = None
    from_var: bool = False
    var_name: str = ""
    desc: str = ""
    
    # Private attributes
    _data_type: Any | None = PrivateAttr(default=None)
    _obj_base_class: Type[BaseModel] | None = PrivateAttr(default=None)

    # Private validation methods (modular validation)
    def _cast_limits(self) -> None:
        data_type = self._data_type
        try: 
            if self.upper_limit is not None: 
                self.upper_limit = data_type(self.upper_limit)
        except: pass

        try: 
            if self.lower_limit is not None:
                self.lower_limit = data_type(self.lower_limit)
        except: pass

    def _cast_allowed_values(self) -> None:
        data_type = self._data_type
        try: self.allowed_values = list(map(data_type, self.allowed_values))
        except: pass

    def _cast_default(self) -> None:
        data_type = self._data_type
        if self.is_list:
            try: self.default = list(map(data_type, self.default))
            except: pass
        else:
            try: self.default = data_type(self.default)
            except: pass

    def _validate_limits(self) -> None:
        data_type = self._data_type
        if self.upper_limit is not None and not isinstance(self.upper_limit, data_type):
            raise TypeError(f"Upper limit has type {type(self.upper_limit)}, expected {data_type}")
        if self.lower_limit is not None and not isinstance(self.lower_limit, data_type):
            raise TypeError(f"Lower limit has type {type(self.lower_limit)}, expected {data_type}")
        if self.upper_limit is not None and self.lower_limit is not None and self.upper_limit < self.lower_limit:
            raise ValueError(f"Upper limit: {self.upper_limit} must be greater than or equal to lower limit: {self.lower_limit}")

    def _validate_allowed_values(self) -> None:
        data_type = self._data_type
        if len(self.allowed_values) > 0 and not all(isinstance(elem, data_type) for elem in self.allowed_values):
            raise TypeError(f"Allowed values have types {[type(elem) for elem in self.allowed_values]}, expected all {data_type}")

    def _validate_default(self) -> None:
        data_type = self._data_type
        if self.default is not None and self.is_list:
            if not isinstance(self.default, list):
                raise TypeError(f"Default is of type {type(self.default)}, expected {type(list)}")
            if not all(isinstance(elem, data_type) for elem in self.default):
                raise TypeError(f"Default has values of types {[type(elem) for elem in self.default]}, expected all {data_type}")
            if self.upper_limit is not None and not all(elem <= self.upper_limit for elem in self.default):
                raise ValueError(f"Default has values of {[type(elem) for elem in self.default]}, expected all below upper limit {self.upper_limit}")
            if self.lower_limit is not None and not all (elem >= self.lower_limit for elem in self.default):
                raise ValueError(f"Default has values of {[type(elem) for elem in self.default]}, expected all above lower limit {self.lower_limit}")
            if len(self.allowed_values) > 0 and not all (elem in self.allowed_values for elem in self.default):
                raise ValueError(f"Default has values of {[type(elem) for elem in self.default]}, expected all values in {self.allowed_values}")
        elif self.default is not None:
            if not isinstance(self.default, data_type):
                raise TypeError(f"Default has type {type(self.default)}, expected {data_type}")
            if self.upper_limit is not None and self.default > self.upper_limit:
                raise ValueError(f"Default has value: {self.default}, expected below upper limit: {self.upper_limit}")
            if self.lower_limit is not None and self.default < self.lower_limit:
                raise ValueError(f"Default has value: {self.default}, expected above lower limit: {self.lower_limit}")
            if len(self.allowed_values) > 0 and self.default not in self.allowed_values:
                raise ValueError(f"Default has value {self.default}, expected one of {self.allowed_values}")

    @model_validator(mode="after")
    def validate(self) -> 'Parameter':
        # Initalize private attributes
        self._init_private_attributes()

        # Cast limits if possible, errors will be caught later
        self._cast_limits()

        # Cast allowed values if possible, errors will be caught later
        self._cast_allowed_values()

        # Cast default value if possible, errors will be caught later
        self._cast_default()

        # Check upper and lower limits
        self._validate_limits()
        
        # Check list of allowed values
        self._validate_allowed_values()
        
        # Check the default value
        self._validate_default()

        return self

    def _init_private_attributes(self) -> None:
        '''
        Description
        -----------
        - Initalize `self._data_type` to the type specified by `self.data_type`
        - Initalize `self._obj_base_class` to `Parameter`
        '''
        typing = {"str": str, "int": int, "float": float, "bool": bool}
        self._data_type = typing[self.data_type]
        self._obj_base_class = Parameter
 
    def _assign_value(self) -> Tuple:
        if self.is_list: value = (List[self._data_type], self.default)
        else: value = (self._data_type | None, self.default)
        return value

    def _assign_model_validators(self) -> Dict[str, Callable]:
        '''
        Description
        -----------
        Constructs a dictionary of validation functions for use within the BaseModel
        subclass returned by the `to_param()` function.

        Return
        ------
        ```
        validator_dict : Dict[str, Callable]
        ```
        The dictionary of validators which will be provided to the subclass 
        '''
        validator_dict = {}
        # Set value and validator functions
        if self.is_list:
            # Handle advanced type casting
            @field_validator("value", mode='before')
            @classmethod
            def validate_type_cast(cls, v):
                try: v = list(map(self._data_type, list(v)))
                except: raise ValueError(f"Type cast conversion for type {type(v)} failed")
                return v
            validator_dict["validate_type_cast"] = validate_type_cast

            # Validate limits and allowed values
            if self.upper_limit is not None:
                # If an upper limit exists, create a validator function for it
                @field_validator('value')
                @classmethod
                def validate_upper_limit(cls, v):
                    if v is not None and not all(elem <= self.upper_limit for elem in v):
                        raise ValueError(f"{self.name} has values: {[elem for elem in v]}, expected below upper limit: {self.upper_limit}")
                    return v
                validator_dict["validate_upper_limit"] = validate_upper_limit
            if self.lower_limit is not None:
                # If a lower limit exists, create a validator function for it
                @field_validator('value')
                @classmethod
                def validate_lower_limit(cls, v):
                    if v is not None and not all(elem >= self.lower_limit for elem in v):
                        raise ValueError(f"{self.name} has values: {[elem for elem in v]}, expected above lower limit: {self.lower_limit}")
                    return v
                validator_dict["validate_lower_limit"] = validate_lower_limit
            if len(self.allowed_values) > 0:
                # If there are allowed values, create a validator function for it
                @field_validator('value')
                @classmethod
                def validate_allowed_values(cls, v):
                    if v is not None and not all(elem in self.allowed_values for elem in v):
                        raise ValueError(f"{self.name} has values: {[elem for elem in v]}, expected all values in: {self.allowed_values}")
                    return v
                validator_dict["validate_allowed_values"] = validate_allowed_values
        else:
            if self.upper_limit is not None:
                # If an upper limit exists, create a validator function for it
                @field_validator('value')
                @classmethod
                def validate_upper_limit(cls, v):
                    if v is not None and v > self.upper_limit:
                        raise ValueError(f"{self.name} has value: {v}, expected below upper limit: {self.upper_limit}")
                    return v
                validator_dict["validate_upper_limit"] = validate_upper_limit
            if self.lower_limit is not None:
                # If a lower limit exists, create a validator function for it
                @field_validator('value')
                @classmethod
                def validate_lower_limit(cls, v):
                    if v is not None and v < self.lower_limit:
                        raise ValueError(f"{self.name} has value: {v}, expected above lower limit: {self.lower_limit}")
                    return v
                validator_dict["validate_lower_limit"] = validate_lower_limit
            if len(self.allowed_values) > 0:
                # If there are allowed values, create a validation function for it
                @field_validator('value')
                @classmethod
                def validate_allowed_values(cls, v):
                    if v is not None and v not in self.allowed_values:
                        raise ValueError(f"{self.name} has value: {v} expected one of: {self.allowed_values}")
                    return v
                validator_dict["validate_allowed_values"] = validate_allowed_values

        # If no validator functions are required, set to None
        if not validator_dict:
            validator_dict = None

        return validator_dict

    def _assign_model_operators(self, model: Type[BaseModel]) -> None:
        '''
        Description
        -----------
        Private class method for overloading specific operators of the BaseModel 
        subclass returned in the `to_param()` class method.

        This method patches in operator functionality to allow for usage of 
        addition, subtraction, multiplication, and other operators directly on 
        objects which are created from the subclass.

        Parameters
        ----------
        ```
        model : Type[BaseModel]
        ```
        The model that is created in the `to_param()` method
        '''
         # Patch in basic binary operators to the model
        def __add__(self, other): return self.value + other
        def __sub__(self, other): return self.value - other
        def __mul__(self, other): return self.value * other
        def __truediv__(self, other): return self.value / other
        def __pow__(self, other): return self.value ** other

        model.__add__ = __add__
        model.__sub__ = __sub__
        model.__mul__ = __mul__
        model.__truediv__ = __truediv__
        model.__pow__ = __pow__

        # Patch in comparison operators to the model
        def __eq__(self, other): return self.value == other
        def __ne__(self, other): return self.value != other
        def __lt__(self, other): return self.value < other
        def __gt__(self, other): return self.value > other
        def __le__(self, other): return self.value <= other
        def __ge__(self, other): return self.value >= other

        model.__eq__ = __eq__
        model.__ne__ = __ne__
        model.__lt__ = __lt__
        model.__gt__ = __gt__
        model.__le__ = __le__
        model.__ge__ = __ge__

        # Patch in assignment operators to the model
        def __isub__(self, other): self.value -= other; return self
        def __iadd__(self, other): self.value += other; return self
        def __imul__(self, other): self.value *= other; return self
        def __ipow__(self, other): self.value **= other; return self

        model.__isub__ = __isub__
        model.__iadd__ = __iadd__
        model.__imul__ = __imul__
        model.__ipow__ = __ipow__


    def to_param(self) -> Type[Parameter]:
        '''
        Description
        -----------
        Converts the parameter model to a parameter class which can be 
        used to create individual parameters. 
        
        ```
        return model : Type[BaseModel]
        ```
        A new model (class) of the parameter model which can be used to create
        objects which enforce the constraints defined in the parameter model.

        Usage
        -----

        ```python
        # Define the voltage model
        voltage_model = ParameterModel(
            name="Voltage",
            data_type="float",
            upper_limit=240.0,
            lower_limit=60.0,
            default=120.0,
            description="Voltage in volts"
        )

        # Build the Voltage class
        Voltage = voltage_model.to_param() # Subclass of Parameter
        
        # Build voltage objects
        v1 = Voltage() # -> has value of 120.0
        v2 = Voltage(value=220.0) # -> has value of 220.0
        v3 = Voltage(value=80.0) # -> has value of 80.0
        ```
        '''

        # Assign the value of the Parameter
        value = self._assign_value()

        # Assign the validator functions for the model
        validator_dict = self._assign_model_validators()

        # Create the dynamic model
        model = create_model(
            self.name,
            value=value,
            from_var=(bool, self.from_var),
            var_name=(str, self.var_name),
            desc=(str, self.desc),
            upper_limit=(self._data_type, self.upper_limit),
            lower_limit=(self._data_type, self.lower_limit),
            __validators__=validator_dict,
            __base__=self._obj_base_class,
        )

        # Assign operators to the model
        self._assign_model_operators(model)

        return model