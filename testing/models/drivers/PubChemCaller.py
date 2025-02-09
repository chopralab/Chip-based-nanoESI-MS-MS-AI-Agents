'''
This is the Module for Pubchem API caller.

This is one of the best Chemical databases. You have to use this database for any queries related to chemical structures, properties, biological activities, substance IDs (SIDs), compound IDs (CIDs), synonyms, descriptions, properties, assay details, or classification nodes.

You have access to specialized PubChem API functions for retrieving chemical and biological information. Whenever you encounter queries related to chemical structures, properties, biological activities, molecular weights, substance IDs (SIDs), compound IDs (CIDs), synonyms, descriptions, properties, assay details, or classification nodes, utilize the provided PubChem functions to fetch accurate and detailed information. Here are some functions you should use based on the type of information requested:oka

Substance IDs (SIDs) from Compound IDs (CIDs):

Function: get_sids_from_cid(inp: str, inp_type: str = 'compound') -> dict
Usage: Retrieves SIDs for given CIDs.
Compound IDs (CIDs) from Substance IDs (SIDs):

Function: get_cids_from_sid(inp: str, inp_type: str = 'substance') -> dict
Usage: Retrieves CIDs for given SIDs.
Synonyms:

Function: get_smiles_from_name(name: str) -> dict
Usage: Retrieves SMILES for a given chemical name.

Function: get_synonym(inp: str, inp_format: str, inp_type: str) -> dict
Usage: Retrieves synonyms for a given identifier.
Descriptions:

Function: get_description(inp: str, inp_format: str, inp_type: str) -> dict
Usage: Retrieves descriptions for a given identifier.
Classification Nodes:

Function: _get_classification_nodes(output_format: str, hnid: str) -> dict
Usage: Retrieves classification nodes for a given hierarchical node identifier (hnid).
Compound Property Table:

Function: get_compound_property_table(inp: str, inp_format: str, inp_type: str, property_list: str) -> dict
Usage: Retrieves a table of properties for a given compound or substance.
Assay Descriptions:

Function: get_assay_description_from_AID(aid: str) -> dict
Usage: Retrieves assay descriptions, protocols, and comments for given assay IDs.
Assay IDs from SMILES:

Function: get_assay_id_from_smiles(smiles: str) -> dict
Usage: Retrieves assay IDs for a given SMILES string of a compound.
Assay Names from Assay IDs:

Function: get_assay_name_from_aid(aid: str) -> dict
Usage: Retrieves names of assays for given assay IDs.
Examples:

Question: "What are the synonyms for caffeine?"
Agent's Action: Use the get_synonym function to retrieve synonyms for caffeine.

Question: "Can you provide the assay description for AID 12345?"
Agent's Action: Use the get_assay_description function to retrieve the description for AID 12345.

Question: "What are the properties of benzene?"
Agent's Action: Use the get_compound_property_table function to retrieve the properties of benzene.

Instructions to the Agent:

Prioritize using the PubChem functions for any query related to chemical substances, their properties, or biological activities.
Avoid using domain knowledge to answer these queries and rather rely on the PubChem API functions.
Ensure the inputs are correctly formatted and any necessary conversions are applied.
Clearly cite PubChem as the source of the information provided to ensure reliability and traceability.

'''
#TODO: modify the docstring and also mention that it should only observe the results and use the and not make up stuff

from uuid import uuid4
from typing import List
from typing import Dict
import pandas as pd
import json
import requests
from ASPIRE_LINQX.ai.tools.constants import PUBCHEM, OUTPUT_FORMAT, PROPERTIES, CACTUS
import pprint

#? Should I shift these two helper functions somwhere else and import them here?
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

def _get_request(url, response_type='json') -> dict:
    """
    This is a helper function to make a GET request to the given URL and return the JSON response. MUST not be used directly.
    """
    response = requests.get(url)
    if response.status_code == 200:
        if response_type == 'json':
            return response.json()
        else:
            return {"response": response.text}
    else: 
        print('Error: \n Staus code: {}\nMessage: {}\n URL created: {}'.format(response.status_code, response.text, url))
        return None  # or return some error indicating that the request URL is not valid


# def get_sids_from_cid(inp:str, inp_type:str='compound') -> dict:
#     '''
#     Function purpose
#     get SIDs (Substance IDs) for a given CID (Compound ID)
    
#     Inputs
#     inp: string representation of a list of CIDs
#     inp_type: 'compound' if inp is a list of CIDs or 'substance' if inp is a list of SIDs

#     returns
#     dict: a dictionary of SIDs for each CID
#     '''
#     try:
#         # inp = _convert_to_string(inp)["l_ids"]
#         inp = inp.replace(" ", "")
#         url = '/'.join([PUBCHEM, inp_type, 'cid', inp, 'sids', OUTPUT_FORMAT])
        
#         return _get_request(url)
#     except Exception as e:
#         print('An error occurred:', e)
#         # return e
#         return None  


# def get_cids_from_sid(inp:str, inp_type:str='substance') -> dict:
#     '''
#     Function purpose
#     get CID (Compound IDs) for a given SID (Substance ID)
    
#     Inputs
#     inp: a string representation of a list of SIDs
#     inp_type: 'compound' if inp is a list of CIDs or 'substance' if inp is a list of SIDs

#     returns
#     dict: a dictionary of CIDs for each SID
#     '''
#     try:
#         # inp = _convert_to_string(inp)["l_ids"]
#         inp = inp.replace(" ", "")
#         url = '/'.join([PUBCHEM, inp_type, 'sid', inp, 'cids', OUTPUT_FORMAT])
    
#         return _get_request(url)
#     except Exception as e:
#         print('An error occurred:', e)
#         # return e
#         return None 
    

def get_cid_from_name(inp: str) -> dict:
    """
    Function purpose
    Get the cid from chemical name

    Inputs
    inp: string representation of a chemical name

    returns
    dict: a dictionary of cids for the given chemical name

    """
    try:
        url = '/'.join([PUBCHEM, 'substance', 'name', inp, 'cids', 'TXT'])
        res = _get_request(url, 'text')
        res = set(res['response'].replace(" ", "").split('\n'))
        return {"response": res}
    except Exception as e:
        print('An error occurred:', e)
        return str(e)

#CACTUS: Name to SMILES
def get_smiles_from_name(inp: str) -> dict:
    '''
    Function purpose
    Get SMILES for a given chemical name. 
    
    Inputs
    inp: string representation of a chemical name
    
    returns
    str: a text representing the SMILES for the given chemical name
    '''
    try:
        url = '/'.join([CACTUS, inp, 'smiles'])
        return _get_request(url, 'text')
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
        inp = inp.replace(" ", "")
        # inp = _convert_to_string(inp)["l_ids"]
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
    inp: string representation of an single chemical name only
    inp_format: string of either of name, sid, cid, smiles 
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance
    
    returns
    dict: a dictionary of descriptions for each identifier
    '''
    try: 
        # inp = _convert_to_string(inp)["l_ids"]
        inp = inp.replace(" ", "")
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
    inp: string representation of an single chemical name or a list of identifiers separated by comma
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
        inp = inp.replace(" ", "")
        url =  '/'.join([PUBCHEM,  inp_type, inp_format, inp, 'property', property_list, OUTPUT_FORMAT])  
        return _get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None    

  
def get_assay_results_for_compund(cid: str, activity_name: str=None) -> dict:
    """
    Function purpose
    Gets all the assay results for a provided compund with an optional filter of assay activity type.

    Inputs:
    cid: a single cid representing a compound
    activity_name (optional): the specific activity type to filter on for example Ki, Kd, IC50, etc

    returns
    dict: a disctionary representation of the assay result table
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/121596705/assaysummary/JSON
    """
    try:
        inp = cid.replace(" ", "")
        url = '/'.join([PUBCHEM, 'compound', 'cid', inp, 'assaysummary', OUTPUT_FORMAT])

        res = _get_request(url)

        columns = res['Table']['Columns']['Column']
        rows = [row['Cell'] for row in res['Table']['Row']]

        df = pd.DataFrame(rows, columns=columns)
        if activity_name:
            df = df[df["Activity Name"] == activity_name]

        return df.to_dict()
    except Exception as e:
        print('An error occurred:', e)
        return None  


def get_assay_description_from_AID(aid: str) -> dict:
    """
    Function purpose
    Get high level Assay description, protocol and comment on the scores for a given assay id.
    
    Inputs
    aid: string representation of a list of assay IDs
    
    returns
    dict: a dictionary of description, protocol and comment for the given assay ID
    """
    try:
        inp = aid.replace(" ", "")
        
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

# TODO: add default number of assay IDs, try with docstring instruction as well.

# def get_assay_id_from_smiles(smiles: str) -> dict:
#     '''
#     Function purpose
#     Gives you the assay ID (aid) for a single smiles string of a compound. 
#     If the user specifies that the item is a substance, then ask the user to enter SMILES for a compound
    
#     Note: An assay is a process of analyzing a compound to determine its composition or quality.
#     This function gives you all the assays that have used the given compound for testing.
    
#     Inputs
#     smiles: smiles string of a compound
    
#     returns
#     str: assay ID for the given smiles string
#     '''
#     try:
#         url = '/'.join([PUBCHEM, 'compound', 'smiles', smiles, 'aids', OUTPUT_FORMAT]) 
#         print(url)
#         res = _get_request(url)
#         return {"AID": res['InformationList']['Information'][0]['AID']}
#     except Exception as e:
#         print('An error occurred:', e)
#         # return e
#         return None   
    
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
        inp = aid.replace(" ", "")
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
    
#TODO:
# Specify List[str] in function signature
# Taking in string in function signature and specify that it is a list of ID's