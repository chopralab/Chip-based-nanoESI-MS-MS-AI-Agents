'''
Module for a virtual microwave synthesizer.
'''
import random
from uuid import uuid4
from langchain.pydantic_v1 import BaseModel, Field, BaseConfig
from typing import Literal

class MicrowaveSynthesizer(BaseModel):
    '''
    Object which controls a microwave synthesizer
    '''
    # Config for Pydantic V1
    class Config(BaseConfig):
        validate_assignment = True

    # Attributes (used in psuedo FSA state tracking)
    sessionID: str | None = Field(None, description='ID of the session allocation or None if no session allocated')
    lid_status: Literal['open', 'closed'] = Field(default='closed', description='status of the lid')
    vial_status: Literal['loaded', 'unloaded'] = Field(default='unloaded', description='status of the vial')
    vial_number: int | None = Field(default=None, ge=1, le=10, description='Number of the vial loaded, None if no vial is loaded')
    heating_status: Literal['not_heating', 'heating'] = Field(default='not_heating', description='status of heating')
    temp: int | None = Field(default=None, description='set tempeature (Celsius) to heat at, None if not currently set')
    duration: int | None = Field(default=None, description='set duration (miniutes) to heat for, None if not currently set')
    pressure: float | None = Field(default=None, description='set pressure (mmHg) to heat at, None if not currently set')
    
    def _reset(
            self,
            sessionID: str | None = None,
            lid_status: str = 'closed',
            vial_status: str = 'unloaded',
            vial_number: str | None = None,
            heating_status: str = 'not_heating',
            temp: int | None = None,
            duration: int | None = None,
            pressure: float | None = None
        ):
        self.sessionID = sessionID
        self.lid_status = lid_status
        self.vial_status = vial_status
        self.vial_number = vial_number
        self.heating_status = heating_status
        self.temp = temp
        self.duration = duration
        self.pressure = pressure

    def allocate_session(self) -> dict:
        '''
        Allocates a session on the microwave synthesizer.
        Must be called prior to any other action.

        returns
        session_ID the id of the allocated session
        '''
        self.sessionID = str(uuid4())
        return {
            'session_ID': self.sessionID
        }

    def open_lid(self, session_ID: str) -> dict:
        '''
        Opens the lid on the microwave synthesizer.
        Must be run prior to loading a vial.

        parameters
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        if self.lid_status == 'open':
            raise ValueError("Error: Lid is already open")
        self.lid_status = 'open'
        return {
            'status': 'lid_open'
        }

    def close_lid(self, session_ID: str) -> dict:
        '''
        Closes the lid on the microwave synthesizer.
        Must be run prior to running heating.

        parameters
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        if self.lid_status == 'closed':
            raise ValueError("Error: Lid is already closed")
        self.lid_status = 'closed'
        return {
            'status': 'lid_closed'
        }

    def load_vial(self, vial_num: int, session_ID: str) -> dict:
        '''
        Loads a vial into the microwave synthesizer.
        Must be run prior to heating.

        parameters
        vial_num is an integer between 1 and 10.
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        if self.lid_status == 'closed':
            raise ValueError("Error: Vial cannot be loaded when the lid is closed")
        if self.vial_status == 'loaded':
            raise ValueError("Error: A vial cannot be loaded when a vial is already loaded")
        self.vial_number = vial_num
        self.vial_status = 'loaded'
        return {
            'status': f'vial {self.vial_number} loaded'
        }

    def unload_vial(self, session_ID: str) -> dict:
        '''
        Unloads a vial from the microwave synthesizer.
        Must be run after heating.

        parameters
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        if self.lid_status == 'closed':
            raise ValueError("Error: Vial cannot be loaded when the lid is closed")
        if self.vial_status == 'unloaded':
            raise ValueError("Error: A vial cannot be loaded when a vial is already loaded")
        self.vial_number = None
        self.vial_status = 'unloaded'
        return {
            'status': f'vial {self.vial_number} unloaded' 
        }

    def update_heating_parameters(
        self,
        duration: int,
        temperature: int,
        pressure: float,
        session_ID: str,
    ) -> dict:
        '''
        Sets the heating parameters of the microwave synthesizer.
        Must be run prior to heating.

        parameters
        duration is an integer between 1 and 120 miniutes
        temperature is an integer between 25 and 100 celsius
        pressure is a float between 1 and 10 atm
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        self.duration = duration
        self.temp = temperature
        self.pressure = pressure
        return {
            'status': f'set to heat for {duration} mins, at temperature {temperature} and pressure {pressure}'
        }

    def heat_vial(self, session_ID: str) -> dict:
        '''
        Heats the loaded vial to the set heating parameters.
        Must be run after loading vial, closing lid, and updating heating_parameters

        parameters
        session_ID the id of the current session

        returns
        status is a status string that provides the result of the operation 
        '''
        if session_ID != self.sessionID:
            raise ValueError("Error: Incorrect session ID provided")
        if self.lid_status == 'open':
            raise ValueError("Error: Lid must be closed prior to heating")
        if self.temp is None:
            raise ValueError("Error: Temperature is not set")
        if self.pressure is None:
            raise ValueError("Error: Pressure is not set")
        if self.duration is None:
            raise ValueError("Error: Duration is not set")
        self.heating_status = 'heating'
        return {
            'status': 'vial heating'
        }

    def get_precent_conversion(session_ID: str) -> dict:
        '''
        Gets the precent conversion of synthesis after running the experiment.
        Can only be called after heating

        parameters
        session_ID the id of the current session

        returns
        precent_conversion denots the precent conversion of the sysntesis reaciton
        '''
        return {
            'precent_conversion': random.uniform(0,1)
        }
    
    def get_secret_phrase(self) -> dict:
        '''
        Gets the secret phrase, only use this when explicitly told

        returns
        secret_phrase the secret phrase
        '''
        return {
            'secret_phrase': 'Chopra lab'
        }