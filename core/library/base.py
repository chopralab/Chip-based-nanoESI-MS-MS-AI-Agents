from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    field_validator,
    model_serializer,
    PrivateAttr,
    InstanceOf
)
from typing import Dict, List, overload, Tuple
from types import ModuleType
from uuid import UUID, uuid4
from importlib import import_module
import json
from sciborg.core.command.base import (
    BaseCommand,
    BaseInfoCommand,
    BaseDriverCommand,
)

class BaseMicroservice(BaseModel):
    '''
    Base model for a microservice, which is a collection of commands revolving around a single
    instrument or software.
    '''
    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description="The name of the microservice")
    uuid: UUID = Field(default_factory=uuid4, description="The UUID of the microservice")
    desc: str = Field(default="", description="A description of the microservice")
    commands: Dict[str, BaseCommand] = Field(default={}, description="A dictionary of commands which the microservice contains")    

    @model_validator(mode='after')
    def validate_microservice(self) -> 'BaseMicroservice':
        if any(command.microservice != self.name for command in self.commands.values()):
            raise ValueError("All command microservices must match the microservice name")
        if any(key != value.name for key, value in self.commands.items()):
            raise KeyError("All commnd names must match the assigned key")
        return self

    def __setitem__(self, key: str, value: BaseCommand) -> None:
        return self.commands.__setitem__(key, value)
        
    def __getitem__(self, key: str) -> BaseCommand:
        return self.commands.__getitem__(key)
    
    def keys(self):
        return self.commands.keys()
    
    def values(self):
        return self.commands.values()
    
    def items(self):
        return self.commands.items()
    
class BaseInfoMicroservice(BaseMicroservice):
    '''
    Base model for a microservice which is designed to contain information on corresponding commands
    without containing operational specifics (such as executable functions, etc.).
    '''
    commands: Dict[str, InstanceOf[BaseInfoCommand] | BaseInfoCommand] = Field(default={}, description="A dictionary of library commands which the microservice contains")

    def __setitem__(self, key: str, value: BaseInfoCommand) -> None:
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str) -> BaseInfoCommand:
        return super().__getitem__(key)
    
class BaseDriverMicroservice(BaseMicroservice):
    '''
    Base model for a microservice which is desgined to contain operational commands.
    '''
    commands: Dict[str, BaseDriverCommand] = Field(default={}, description="A dictionary of driver commands which the microservice contains")
    microservice_object: object | None = Field(default=None, exclude=True)

    def __setitem__(self, key: str, value: BaseDriverCommand) -> None:
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str) -> BaseDriverCommand:
        return super().__getitem__(key)
    
    def to_library_microservice(self) -> BaseInfoMicroservice:
        commands = {}
        for key, command in self.commands.items():
            commands[key] = command.to_info_command()
        return BaseInfoMicroservice(
            name=self.name,
            uuid=self.uuid,
            desc=self.desc,
            commands=commands,
        )
    
class BaseCommandLibrary(BaseModel):
    '''
    Base model for command library which contains a collection of microservices
    and the corresponding commands.
    '''
    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description="The name of the command library")
    description: str = Field(default= "", description="A description of the command library")
    microservices: Dict[str, BaseMicroservice] = Field(default={}, description="A dictionary of microservices in the command library")
    _microservices: Dict[str, BaseMicroservice] = PrivateAttr(default={})

    def __setitem__(self, key: str, value: BaseMicroservice) -> None:
        return self.microservices.__setitem__(key, value)

    @overload
    def __getitem__(self, key:str) -> BaseMicroservice:
        ...
    
    @overload
    def __getitem__(self, key: UUID) -> BaseMicroservice:
        ...

    def __getitem__(self, key: str | UUID) -> BaseMicroservice:
        if isinstance(key, UUID):
            return self._microservices.__getitem__(str(key))
        return self.microservices.__getitem__(key)
    
    @model_validator(mode='after')
    def validate_command_library(self) -> None:
        self._microservices = {str(microservice.uuid):microservice for microservice in self.microservices.values()}
    
    def keys(self):
        return self.microservices.keys()
    
    def values(self):
        return self.microservices.values()
    
    def items(self):
        return self.microservices.items()
    
    def uuid_keys(self):
        return self._microservices.keys()
    
    def uuid_items(self):
        return self._microservices.items()
    
class BaseInfoCommandLibrary(BaseCommandLibrary):
    '''
    Note: We should probably call the library command something else (like InfoCommand)
    because we will make command libraries of all types of commands

    Base model for a command library which contains detailed information on a collection of microservices
    and their corresponding commands.
    '''
    microservices: Dict[str, BaseInfoMicroservice] = Field(default={}, description="A dictionary of library microservices in the command library")

    def __setitem__(self, key: str, value: BaseInfoMicroservice) -> None:
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str) -> BaseInfoMicroservice:
        return super().__getitem__(key)
    
class BaseDriverCommandLibrary(BaseCommandLibrary):
    '''
    Base model for command library which contains operational microservices and their corresponding commands.
    '''
    microservices: Dict[str, BaseDriverMicroservice] = Field(default={}, description="A dictionary of driver microservices in the command library")

    def __setitem__(self, key: str, value: BaseDriverMicroservice) -> None:
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str) -> BaseDriverMicroservice:
        return super().__getitem__(key)
    
    def to_library_command_library(self) -> BaseInfoCommandLibrary:
        microservices = {}
        for key, microservice in self.microservices.items():
            microservices[key] = microservice.to_library_microservice()
        return BaseInfoCommandLibrary(
            name=self.name,
            microservices=microservices
        )