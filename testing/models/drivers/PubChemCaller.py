'''
Module for Pubchem API caller.
'''

from uuid import uuid4
from typing import List
from typing import Dict

import requests
from sciborg.ai.tools.constants import PUBCHEM, OUTPUT_FORMAT, PROPERTIES
import pprint

def _convert_to_string(l_ids) -> dict:
    """
    This is a helper function to convert a list of IDs to a string of IDs separated by commas. MUST not be used directly.
    """
    if isinstance(l_ids, list):
        l_ids = [str(l) for l in l_ids] #Do we need this if the LLM is calling it?
        l_ids = ','.join(l_ids)
    else: 
        l_ids = str(l_ids)
    return {"l_ids": l_ids}

def _get_request(url) -> dict:
    """
    This is a helper function to make a GET request to the given URL and return the JSON response. MUST not be used directly.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else: 
        print('Error: \n Staus code: {}\nMessage: {}\n URL created: {}'.format(response.status_code, response.text, url))
        return None  # or return some error indicating that the request URL is not valid


def get_sids_from_cid(inp:str, inp_type:str='compound') -> dict:
    '''
    Function purpose
    get SIDs (Substance IDs) for a given CID (Compound ID)
    
    Inputs
    inp: string representation of a list of CIDs
    inp_type: 'compound' if inp is a list of CIDs or 'substance' if inp is a list of SIDs

    returns
    dict: a dictionary of SIDs for each CID
    '''
    try:
        # inp = _convert_to_string(inp)["l_ids"]
        
        url = '/'.join([PUBCHEM, inp_type, 'cid', inp, 'sids', OUTPUT_FORMAT])
        
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None  


def get_cids_from_sid(inp:str, inp_type:str='substance') -> dict:
    '''
    Function purpose
    get CID (Compound IDs) for a given SID (Substance ID)
    
    Inputs
    inp: a string representation of a list of SIDs
    inp_type: 'compound' if inp is a list of CIDs or 'substance' if inp is a list of SIDs

    returns
    dict: a dictionary of CIDs for each SID
    '''
    try:
        # inp = _convert_to_string(inp)["l_ids"]
        url = '/'.join([PUBCHEM, inp_type, 'sid', inp, 'cids', OUTPUT_FORMAT])
    
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    

def get_synonym(inp: str, inp_format: str, inp_type: str) -> dict:
    '''
    Function purpose
    Get Synonym of a substance or compound.
    
    Inputs
    inp: string representation of a list of identifiers
    inp_format: string which can be either of name, sid, cid, smiles
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance
    
    returns
    dict: a dictionary of synonyms for each identifier
    '''
    try: 
        inp = _convert_to_string(inp)["l_ids"]
        url = '/'.join([PUBCHEM, inp_type, inp_format, inp, 'synonyms', OUTPUT_FORMAT])
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    

def get_description(inp: str, inp_format: str, inp_type: str) -> dict:
    '''
    Function purpose 
    Get description of a substance or a compound, for assay description, use get_assay_description() instead

    Inputs
    inp: string representation of a list of identifiers
    inp_format: string of either of name, sid, cid, smiles 
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance
    
    returns
    dict: a dictionary of descriptions for each identifier
    '''
    try: 
        inp = _convert_to_string(inp)["l_ids"]
        url = '/'.join([PUBCHEM, inp_type, inp_format, inp, 'description', OUTPUT_FORMAT])
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    

def _get_classification_nodes(output_format , hnid) -> dict:
    '''
    Function purpose
    Get classification nodes for a given hierarchical node identifier (hnid).

    Inputs
    The output forma
    
    returns
    dict: a dictionary of classification nodes for the given hnid
    '''
    try:
        url =  '/'.join([PUBCHEM, 'classification/hnid', str(hnid), output_format, OUTPUT_FORMAT])
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 

   
def get_compound_property_table(inp: str, inp_format: str, inp_type: str, property_list: str) -> Dict[str, str | int]:
    # works for name, cid, cids, smiles
    f'''
    Function purpose    
    Get a table of properties for a given compound or substance.
    
    Inputs
    inp: string representation of list of identifiers seperated by commas
    inp_format: one of name, sid, cid, smiles corresponding to the identifiers in inp
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance
    property_list: string representation of list of properties seperated by commas. Must only include from the {PROPERTIES} list
    
    returns
    dict: a dictionary of the key of property name and value of property value for each property in the property_list
    '''
    try:
        # if len(property_list) == 0 or not set(property_list).issubset(set(PROPERTIES)):
        #     raise ValueError("Invalid property list")
        # inp = _convert_to_string(inp)["l_ids"]
        # property_list = _convert_to_string(property_list)["l_ids"]
        print(property_list)
        url =  '/'.join([PUBCHEM,  inp_type, inp_format, inp, 'property', property_list, OUTPUT_FORMAT])  
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None    

  
def get_assay_description(aid: int) -> dict:
    """
    Function purpose
    Get Assay description, protocol and comment on the scores for a given assay id.
    
    Inputs
    aid: assay ID
    
    returns
    dict: a dictionary of description, protocol and comment for the given assay ID
    """
    try:
        inp = _convert_to_string(aid)["l_ids"]
        url = '/'.join([PUBCHEM, "assay", "aid", inp, 'description', OUTPUT_FORMAT])
        print(url)
        res = _get_request(url)
        # pp.pprint(res)
        specific_pairs = {}
        res = res['PC_AssayContainer'][0]['assay']['descr']

        # Iterate over the dictionary
        for key in ["description", "protocol", "comment"]:
        # Check if the key exists in the dictionary
            if key in res:
                # Add the key-value pair to the specific_pairs dictionary
                specific_pairs[key] = res[key]

        return specific_pairs
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None   

def get_assay_id_from_smiles(smiles: str) -> dict:
    '''
    Function purpose
    Gives you the assay ID (aid) for a single smiles string of a compound. 
    If the user specifies that the item is a substance, then ask the user to enter SMILES for a compound
    
    Note: An assay is a process of analyzing a compound to determine its composition or quality.
    This function gives you all the assays that have used the given compound for testing.
    
    Inputs
    smiles: smiles string of a compound
    
    returns
    str: assay ID for the given smiles string
    '''
    try:
        url = '/'.join([PUBCHEM, 'compound', 'smiles', smiles, 'aids', OUTPUT_FORMAT]) 
        print(url)
        res = _get_request(url)
        return {"AID": res['InformationList']['Information'][0]['AID']}
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None   
    
def get_assay_name_from_aid(aid: str) -> dict:
    """
    Function purpose
    Gives a dictionary of names for each assay ID (aid)

    Inputs
    aid: string representation of a list of assay IDs
    
    returns
    str: a dictionary of names for each assay ID
    """
    try:
        inp = _convert_to_string(aid)["l_ids"]
        url = '/'.join([PUBCHEM, 'assay', 'aid', inp, 'description', OUTPUT_FORMAT]) 
        res = _get_request(url)
        # pp.pprint(res)
        res = res['PC_AssayContainer']
        names = dict()
        for i, desc in enumerate (res):
            id = str(desc['assay']['descr']['aid']['id'])
            name = desc['assay']['descr']['name']
            names[id] = name
            print('names', names[id])
        return names
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None