# from langchain.pydantic_v1 import BaseModel, PositiveInt, root_validator, Field
from pydantic import BaseModel as BaseModelV2, Field, PositiveInt
from typing import Literal, List, Dict, Any

from sciborg_dev.core.parameter.base import ParameterModel, ValueType

class ParameterModelSchemaV1(BaseModelV2):
    # Parameter model attributes
    name: str = Field(description='The name of the parameter')
    data_type: Literal["str", "int", "float"] = Field(description='The data type of the parameter')
    precision: Literal[-1] | PositiveInt = Field(default=-1, description='The percision of the parameter, do not assign if unspecified')
    upper_limit: int | str | float | None = Field(default=None, description='The upper limit of the parameter, do not assign if unspecified, must be above lower limit')
    lower_limit: int | str | float | None = Field(default=None, description='The lower limit of the parameter, do not assign if unspecified, must be below upper limit')
    allowed_values: List[int | str | float] = Field(default=[], description='A list of allowed values, do not assign if unspecified')
    is_optional: bool = Field(default=False, description='True if specified as an optional parameter, false otherwise')
    is_list: bool = Field(default=False, description='True is specified that the parameter is a list, false otherwise')
    default: ValueType | None = Field(default=None, description='Set a default value if specified, must be between the upper and lower limit of the parameter')
    from_var: bool = Field(default=False, description='True if specified that the parameter is read from a variable, false otherwise')
    var_name: str = Field(default='', description='The name of the varaible which the parameter will be read from, do not assign if from_var is false')
    desc: str = Field(default='', description='A short description of the parameter')

class ParameterSchemaV1(BaseModelV2):
    value: ValueType = ""
    # desc: str = ""
    from_var: bool = False
    var_name: str = ""