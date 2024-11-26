'''
Module for a virtual microwave synthesizer.
'''
import random
from uuid import uuid4

def allocate_session() -> dict:
    '''
    Allocates a session on the microwave synthesizer.
    Must be called prior to any other action.

    returns
    session_ID the id of the allocated session
    '''
    return {
        'session_ID': str(uuid4())
    }

def open_lid(session_ID: str) -> dict:
    '''
    Opens the lid on the microwave synthesizer.
    Must be run prior to loading a vial.

    parameters
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
    return {
        'status': 'lid_open'
    }

def close_lid(session_ID: str) -> dict:
    '''
    Closes the lid on the microwave synthesizer.
    Must be run prior to running heating.

    parameters
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
    return {
        'status': 'lid_closed'
    }

def load_vial(vial_num: int, session_ID: str) -> dict:
    '''
    Loads a vial into the microwave synthesizer.
    Must be run prior to heating.

    parameters
    vial_num is an integer between 1 and 10.
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
    return {
        'status': f'vial {vial_num} loaded'
    }

def unload_vial(session_ID: str) -> dict:
    '''
    Unloads a vial from the microwave synthesizer.
    Must be run after heating.

    parameters
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
    return {
        'status': 'current vial unloaded' 
    }

def update_heating_parameters(
    duration: int,
    temperature: int,
    pressure: float,
    session_ID: str,
) -> dict:
    '''
    Sets the heating parameters of the microwave synthesizer.
    Must be run prior to heating.

    parameters
    duration is an integer between 1 and 60 miniutes
    temperature is an integer between 25 and 250 celsius
    pressure is a float between 1 and 10 atm
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
    return {
        'status': f'set to heat for {duration} mins, at temperature {temperature} and pressure {pressure}'
    }

def heat_vial(session_ID: str) -> dict:
    '''
    Heats the loaded vial to the set heating parameters.
    Must be run after loading vial, closing lid, and updating heating_parameters

    parameters
    session_ID the id of the current session

    returns
    status is a status string that provides the result of the operation 
    '''
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