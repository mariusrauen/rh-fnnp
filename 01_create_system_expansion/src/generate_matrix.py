#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import requests
import logging
from dataclasses import dataclass 

@dataclass
class CMProcess:
    process_id: int
    product: str
    product_coeff: float
    product_raw_material_id: int
    coproducts: list[str]
    coproducts_coeff: list[float]
    coproducts_raw_material_id: list[int]
    educts: list[str]
    educts_coeff: list[str]
    educts_raw_material_id: list[int]


def get_compound_cas_from_smile_or_name(query):
    """Get a compound's CAS number using the name or smile."""
    base_url = "https://commonchemistry.cas.org/api/"

    # Set the API endpoint and parameters
    endpoint = "search?q="
    
    # Send the GET request to the CAS API
    response = requests.get(base_url + endpoint + query)
    #print(base_url + endpoint + can_smile)
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        logging.debug(f"Working on '{query}'")
        logging.debug(data['results'])
        if data['results'] == []:
            return 'Name not Found'
        else:
            return data['results'][0]['rn']
    else:
        print("Error:", response.status_code)
        return None

def get_compound_info(cas_id):
    """Get a compound's molecular data using a CAS_ID."""
    if cas_id == 'Name not Found':
        return 'Name not Found'
    else:
        base_url = "https://commonchemistry.cas.org/api/"

        # Set the API endpoint and parameters
        endpoint = "detail?cas_rn="

        # Send the GET request to the CAS API
        response = requests.get(base_url + endpoint + cas_id)
        print(base_url + endpoint + cas_id)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            return None
  
def get_chemical_details(cas_api_data):
    """Get CAS-Nr., chemical formular (with deleted html expressions), molmass and name from previous query."""
    if cas_api_data == 'Name not Found':
        return {'name':'Name not Found', 'CAS-Nr.':'Name not Found','chemical formular':'Name not Found','molecular mass':'Name not Found', 'SMILES':'Name not Found'}
    else:
        return {'name':cas_api_data['name'].lower(), 'CAS-Nr.':cas_api_data['rn'],'chemical formular':cas_api_data['molecularFormula'].replace('<sub>','').replace('</sub>',''),'molecular mass':cas_api_data['molecularMass'], 'SMILES':cas_api_data['canonicalSmile']}

def get_cas_data(queries):
    """Given a list of names this function returns a pandas dataframe containing relevant chemical data."""
    df = pd.DataFrame()
    for query in queries:
        df = pd.concat([df,pd.DataFrame(get_chemical_details(get_compound_info(get_compound_cas_from_smile_or_name(query))),index=[query])], axis=0)
    return df

def get_cm_db()->pd.DataFrame:
    """Read in Molecule and Metadata of chemicals in CM xls files and generates rdkit mols and generate morgan fingerprints with radius 3 with a length of 2048 bits for all SMILES deposited in the database
    this is added as a 'morgan_fp' column in the database. chemical fingerprint."""

    path_included_chemicals = "/mnt/c/Users/Jonas/Carbon Minds GmbH/CM_Documents - Dokumente/09 cm_chemicals database code/00_DatabaseGeneration/02_techModels/IncludedChemicals.xlsx"
    layer1 = pd.read_excel(path_included_chemicals, sheet_name='LAYER1')
    layer2 = pd.read_excel(path_included_chemicals, sheet_name='LAYER2')
    layer3 = pd.read_excel(path_included_chemicals, sheet_name='LAYER3')
    ecoinvent = pd.read_excel(path_included_chemicals, sheet_name='ECOINVENT')
    all_cm_chemicals = pd.concat([layer1,layer2,layer3,ecoinvent])
    chemicals_meta_data = pd.read_excel("/mnt/c/Users/Jonas/Carbon Minds GmbH/CM_Documents - Dokumente/09 cm_chemicals database code/00_DatabaseGeneration/00_inputData/meta_data_flows.xlsx")
    all_cm = pd.merge(all_cm_chemicals, chemicals_meta_data, left_on='Included chemicals', right_on='name')
    #PandasTools.AddMoleculeColumnToFrame(all_cm,'SMILES CODE','Molecule')
    #fpgen = AllChem.GetMorganGenerator(radius=3)
    #all_cm['morgan_fp'] = all_cm['Molecule'].apply(lambda x: fpgen.GetFingerprint(x) if x != None else x)
    #all_cm['got_cas'] = all_cm.CAS.apply(lambda x: 1 if x!='unspecific' else 0)
    return all_cm

def is_cansmile_in_cm_db(canonical_smile: str, db_file: pd.DataFrame) -> bool:
    """Is the canonical smiles (produced by cas)?"""
    return canonical_smile in db_file['CAS'].unique()

def is_name_in_cm_db(name: str, db_file: pd.DataFrame) -> bool:
    """Is the plain name in the database?"""
    return name in db_file['name'].unique()

def check_mols_not_in_list(list1:list[str], list2:list[str])-> list[str]:
    """ Check if the list of molecule names of list1 is in the second list of strings."""
    for i in list1: 
        if i not in list2:
            print(f'{i} is not in list2!')


def get_all_mols(df: pd.DataFrame)->list[str]:
    """Append all molecule names mentioned in the raw materials and co product columns."""
    allmols = []
    allmols.append(df['main flow'].to_list())
    allmols.append(df['raw material 1'].to_list())
    allmols.append(df['raw material 2'].to_list())
    allmols.append(df['raw material 3'].to_list())
    allmols.append(df['co-product 1'].to_list())
    allmols.append(df['co-product 2'].to_list())
    allmols.append(df['co-product 3'].to_list())
    flatten_list_with_nan = [mol for sublist in allmols for mol in sublist]
    flatten_list = [mol for mol in flatten_list_with_nan if str(mol)!='nan']
    return list(set(flatten_list))

def generate_processes_list_from_reference_sheet_and_raw_material_id(path) -> list[CMProcess]:
    """This Function generates a list that contains instances of the CMProcess class.
    
    Keyword arguments:
    path -- path to reaction extension layer excel file

    """
    raw_material_id = pd.read_excel(path, sheet_name='raw_material_id')
    reference = pd.read_excel(path, sheet_name='reference')
    process = pd.read_excel(path, sheet_name='process_id')
    reference_with_process = pd.merge(reference, process, left_on='main flow', right_on='main flow')
    processes = []
    for index, row in reference_with_process.iterrows():
        # Get info for 
        educts = []
        educts.append(row['raw material 1'])
        educts.append(row['raw material 2'])
        educts.append(row['raw material 3'])
        educts.append(row['raw material 4'])
        educts = [x for x in educts if str(x) != 'nan']
        educts_coeff = []
        educts_coeff.append(-row['coefficient 1'])
        educts_coeff.append(-row['coefficient 2'])
        educts_coeff.append(-row['coefficient 3'])
        educts_coeff.append(-row['coefficient 4'])
        educts_coeff = [x for x in educts_coeff if str(x) != 'nan']
        educts_raw_material_id = []
        missing_materials = []
        for educt in educts:
            if not raw_material_id.loc[raw_material_id['raw materials']==educt].empty :
                educts_raw_material_id.append(raw_material_id.loc[raw_material_id['raw materials']==educt].iloc[0,0])
            else:
                print(f"{educt} is missing in the raw_materials list!")
                missing_materials.append(educt)
        #repeat for coproducts
        coproducts = []
        coproducts.append(row['co-product 1'])
        coproducts.append(row['co-product 2'])
        coproducts.append(row['co-product 3'])
        coproducts.append(row['co-product 4'])
        coproducts = [x for x in coproducts if str(x) != 'nan']
        coproducts_coeff = []
        coproducts_coeff.append(row['coefficient 1.1'])
        coproducts_coeff.append(row['coefficient 2.1'])
        coproducts_coeff.append(row['coefficient 3.1'])
        coproducts_coeff.append(row['coefficient 4.1'])
        coproducts_coeff = [x for x in coproducts_coeff if str(x) != 'nan']
        coproducts_raw_material_id = []
        coproducts_missing_materials = []
        for coproduct in coproducts:
            if not raw_material_id.loc[raw_material_id['raw materials']==coproduct].empty :
                coproducts_raw_material_id.append(raw_material_id.loc[raw_material_id['raw materials']==coproduct].iloc[0,0])
            else:
                print(f"{coproduct} is missing in the raw_materials list!")
                missing_materials.append(coproduct) 
        product = row['main flow']
        process_id = row['process_id']
        product_coeff = row['main flow coefficient']
        
        if not raw_material_id.loc[raw_material_id['raw materials']==product].empty :
                product_raw_material_id = raw_material_id.loc[raw_material_id['raw materials']==product].iloc[0,0]
        else:
            print(f"\x1b[31m\"{product} is missing in the raw_materials list!\"\x1b[0m")
            missing_materials.append(product) 
        
        processes.append(CMProcess(process_id=process_id, product=product, product_coeff=product_coeff, product_raw_material_id=product_raw_material_id,
                  educts=educts, educts_coeff=educts_coeff, educts_raw_material_id=educts_raw_material_id,
                  coproducts=coproducts, coproducts_coeff=coproducts_coeff, coproducts_raw_material_id=coproducts_raw_material_id))
                  
        print(product, process_id)
        print(educts,educts_coeff,educts_raw_material_id)
        print(coproducts,coproducts_coeff,coproducts_raw_material_id)
    
    return processes 

def generate_matrix_from_list_of_processes(processes: list[CMProcess]) -> pd.DataFrame:
    """Given a list of processes this function generates a pandas dataframe containing the A matrix precursor.""" 
    matrix  = pd.DataFrame({'raw_material_id':[0,1],'process_id':[-1,-1], 'coefficient':[-1.2,-2]})
    for process in processes:
        matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.product_raw_material_id,'process_id':process.process_id,'coefficient':process.product_coeff}, index=[0])])
        if len(process.coproducts)==1:
            print(process.process_id, process.coproducts)
            matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.coproducts_raw_material_id[0],'process_id':process.process_id,'coefficient':process.coproducts_coeff[0]}, index=[0])])
        if len(process.coproducts)>1:
            for idx, coproduct in enumerate(process.coproducts):
                matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.coproducts_raw_material_id[idx],'process_id':process.process_id,'coefficient':process.coproducts_coeff[idx]}, index=[0])])
        if len(process.educts)==1:
            print(process.process_id, process.educts)
            matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.educts_raw_material_id[0],'process_id':process.process_id,'coefficient':process.educts_coeff[0]}, index=[0])])
        if len(process.educts)>1:
            for idx, coproduct in enumerate(process.educts):
                matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.educts_raw_material_id[idx],'process_id':process.process_id,'coefficient':process.educts_coeff[idx]}, index=[0])])   
    return matrix

def remove_water_from_processes(processes: list[CMProcess]) -> None:
    ''' This function removes waters from the processes, as they are excluded from allocation in a later step anyways.

    BUT THERE IS STILL A BUG IF THERE ARE MULITPLE WATERS IN THE COPRODUCTS?!
    I QUICKFIXED IT WITH A SECOND LOOP BUT IT DOES NOT APPEAR TO FIX THE PROBLEM '''

    for process in processes:
        if 'water' in process.coproducts:
            water_index = process.coproducts.index('water')
            process.coproducts.remove('water')
            del process.coproducts_coeff[water_index]
            print("Removed water from a Process!")
    print("Round 1")
    for process in processes:
            if 'water' in process.coproducts:
                water_index = process.coproducts.index('water')
                process.coproducts.remove('water')
                del process.coproducts_coeff[water_index]
                print("Removed water from a Process!")

def main(path: str, output:str) -> None:
    '''Generate a matrix from the reference sheet xlsx. 

       1. Generate CMProcesses containing reaction data. 
       2. Remove water from processes.
       3. Write excel file into xlsx folder.'''

    processes = generate_processes_list_from_reference_sheet_and_raw_material_id(path)
    remove_water_from_processes(processes)
    matrix = generate_matrix_from_list_of_processes(processes)
    matrix.to_excel(output, index=None)
    print(output)


if __name__ == '__main__':
    path = '../xlsx/reaction_extension_layer.xlsx'
    output = f"{path.replace('.xlsx','_matrix.xlsx')}"
    main(path, output)
