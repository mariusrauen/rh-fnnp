#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import numpy as np
import requests
import logging
from dataclasses import dataclass 
from pathlib import Path
from typing import Union
from shutil import copyfile 
from configparser import ConfigParser

def read_in_config() -> ConfigParser:
    """This function reads in the parameters specified in the config.ini file."""
    config = ConfigParser()
    config.read('config.ini')
    return config

@dataclass
class CMProcess:
    """Captures all necessary information for a single reaction process needed for 
       initial matrix generation."""
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
        logging.debug(base_url + endpoint + cas_id)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logging.info("Error:", response.status_code)
            return None
  
def get_chemical_details(cas_api_data):
    """Get CAS-Nr., chemical formular (with deleted html expressions), molmass and name from previous query."""
    if cas_api_data == 'Name not Found':
        return {'name':'Name not Found', 'CAS-Nr.':'Name not Found','chemical formular':'Name not Found','molecular mass':'Name not Found', 'SMILES':'Name not Found'}
    else:
        return {'name':cas_api_data['name'].lower(), 'CAS-Nr.':cas_api_data['rn'],'chemical formular':cas_api_data['molecularFormula'].replace('<sub>','').replace('</sub>',''),'molecular mass':cas_api_data['molecularMass'], 'SMILES':cas_api_data['canonicalSmile']}

def get_cas_data(queries: list[str]) -> pd.DataFrame:
    """Given a list of names this function returns a pandas dataframe containing relevant chemical data."""
    df = pd.DataFrame()
    for query in queries:
        df = pd.concat([df,pd.DataFrame(get_chemical_details(get_compound_info(get_compound_cas_from_smile_or_name(query))),index=[query])], axis=0)
    return df

def get_cm_db(included_chemicals_master: Path, meta_data_flows_master: Path)->pd.DataFrame:
    """Read in Molecule and Metadata of chemicals in CM xls files and generates rdkit mols and generate morgan fingerprints with radius 3 with a length of 2048 bits for all SMILES deposited in the database
    this is added as a 'morgan_fp' column in the database. chemical fingerprint."""

    path_included_chemicals = included_chemicals_master
    layer1 = pd.read_excel(path_included_chemicals, sheet_name='LAYER1')
    layer2 = pd.read_excel(path_included_chemicals, sheet_name='LAYER2')
    layer3 = pd.read_excel(path_included_chemicals, sheet_name='LAYER3')
    ecoinvent = pd.read_excel(path_included_chemicals, sheet_name='ECOINVENT')
    all_cm_chemicals = pd.concat([layer1,layer2,layer3,ecoinvent])
    chemicals_meta_data = pd.read_excel(meta_data_flows_master)
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

def generate_processes_list_from_reference_sheet_and_raw_material_id(path: Path) -> list[CMProcess]:
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
                  
        logging.DEBUG(product, process_id)
        logging.DEBUG(educts,educts_coeff,educts_raw_material_id)
        logging.DEBUG(coproducts,coproducts_coeff,coproducts_raw_material_id)
    
    return processes 

def generate_matrix_from_list_of_processes(processes: list[CMProcess]) -> pd.DataFrame:
    """Given a list of processes this function generates a pandas dataframe containing the A matrix precursor.""" 
    matrix  = pd.DataFrame({'raw_material_id':[0,1],'process_id':[-1,-1], 'coefficient':[-1.2,-2]})
    for process in processes:
        matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.product_raw_material_id,'process_id':process.process_id,'coefficient':process.product_coeff}, index=[0])])
        if len(process.coproducts)==1:
            logging.DEBUG(process.process_id, process.coproducts)
            matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.coproducts_raw_material_id[0],'process_id':process.process_id,'coefficient':process.coproducts_coeff[0]}, index=[0])])
        if len(process.coproducts)>1:
            for idx, coproduct in enumerate(process.coproducts):
                matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.coproducts_raw_material_id[idx],'process_id':process.process_id,'coefficient':process.coproducts_coeff[idx]}, index=[0])])
        if len(process.educts)==1:
            logging.DEBUG(process.process_id, process.educts)
            matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.educts_raw_material_id[0],'process_id':process.process_id,'coefficient':process.educts_coeff[0]}, index=[0])])
        if len(process.educts)>1:
            for idx, coproduct in enumerate(process.educts):
                logging.DEBUG(process.process_id, process.educts)
                matrix = pd.concat([matrix, pd.DataFrame({'raw_material_id':process.educts_raw_material_id[idx],'process_id':process.process_id,'coefficient':process.educts_coeff[idx]}, index=[0])])   
    return matrix

def remove_water_from_processes(processes: list[CMProcess]) -> None:
    ''' This function removes waters from the processes, as they are excluded from allocation in a later step anyways.

    BUT THERE IS STILL A BUG IF THERE ARE MULITPLE WATERS IN THE COPRODUCTS?!
    I QUICKFIXED IT WITH A SECOND LOOP BUT IT DOES NOT APPEAR TO FIX THE PROBLEM '''

    for process in processes:
        if 'water' in process.coproducts:
            water_index = process.coproducts.index('water')
            del process.coproducts[water_index]
            del process.coproducts_coeff[water_index]
            del process.coproducts_raw_material_id[water_index]
            logging.debug("Removed water from a Process!")

def get_header_for_reaction_extension_layer()->list[str]:
    return(['raw_material_id', 'raw materials', 'CAS-Nr.', 'chemical formular', 'molecular mass','category', '[unit choice]','name','SMILES','comment'])

def get_header_for_meta_data_flows_master_file()->list[str]:
    return(['name', 'category', 'concentration/purity', 'CAS-Nr.', 'unit(choice)', '[unit choice]', 'Market price', 'HHV', 'LHV', 'chemical formular[C2C4 format]', 
            'location(choice)', 'exact location', 'comments', 'SMILES CODE', 'molecular mass', 'flowCategory', 'flowSubcategory'])

def get_data_for_reference_sheet(input_path: Path, sheet_name: str | int, included_chemicals_master: Path | str, meta_data_flows_master_file: Path | str) -> pd.DataFrame:
    raw_mats = pd.read_excel(input_path, sheet_name=sheet_name)
    all_cm_md = pd.read_excel(meta_data_flows_master_file)
    result = (
        pd.merge(raw_mats, all_cm_md, how='left', left_on='raw materials', right_on= 'name', validate='1:1')
    )
    result['CAS-Nr.'] = result['CAS-Nr._x'].fillna(result['CAS-Nr._y'])
    result['chemical formular'] = result['chemical formular'].fillna(result['chemical formular[C2C4 format]'])
    result['molecular mass'] = result['molecular mass_x'].fillna(result['molecular mass_y'])
    result['SMILES'] = result['SMILES'].fillna(result['SMILES CODE'])
    result['[unit choice]'] = 'kg'
    result.loc[0:1,'[unit choice]'] = 'MJ'
    result['name'] = result['raw materials']
    result['category'] = result['category_y']
    result['raw_material_id'] = list(range(len(result)))
    result = result.loc[:,get_header_for_reaction_extension_layer()]
    print(result)
    return result

def get_data_for_addition_to_meta_data_flows_master_file(input_path: Path, sheet_name: str | int, included_chemicals_master: Path | str,
                                                         meta_data_flows_master_file: Path | str) -> pd.DataFrame:
    raw_mats = pd.read_excel(input_path, sheet_name=sheet_name)
    all_cm = get_cm_db(included_chemicals_master, meta_data_flows_master_file)
    result = (
        pd.merge(raw_mats, all_cm, how='left', left_on='raw materials', right_on= 'Included chemicals', validate='1:1', indicator=True)
    )
    result = result.loc[result['_merge'] == 'left_only']
    result['CAS-Nr.'] = result['CAS-Nr._x'].fillna(result['CAS-Nr._y'])
    result['chemical formular[C2C4 format]'] = result['chemical formular'].fillna(result['chemical formular[C2C4 format]'])
    result['molecular mass'] = result['molecular mass_x'].fillna(result['molecular mass_y'])
    result['SMILES CODE'] = result['SMILES'].fillna(result['SMILES CODE'])
    result['[unit choice]'] = 'kg'
    result['name'] = result['raw materials']
    result['unit(choice)'] = 'Mass'
    result['concentration'] = np.nan
    result['HHV'] = np.nan
    result['LHV'] = float(0)
    result['flowCategory'] = 'materials production'
    result['category'] = 1
    result['flowSubcategory'] = 'chemical'
    result = result.loc[:, get_header_for_meta_data_flows_master_file()]
    print(result)
    print(result.columns)
    return result

def get_data_for_addition_to_included_chemicals_file(input_path: Path, included_chemicals_master: Path | str,
                                                         meta_data_flows_master_file: Path | str) -> pd.DataFrame:
    process_id = pd.read_excel(input_path, sheet_name='process_id')
    all_cm = get_cm_db(included_chemicals_master, meta_data_flows_master_file)
    result = (
        pd.merge(process_id, all_cm, how='left', left_on='main flow', right_on= 'Included chemicals', validate='1:1', indicator=True)
    )
    result = result.loc[result['_merge'] == 'left_only']
    result = pd.merge(result, pd.read_excel(input_path, sheet_name='raw_material_id').loc[:,['raw materials', 'CAS-Nr.']], how='left', left_on='main flow', right_on='raw materials')
    print(result)
    result['Process'] = result['process']
    result['Included chemicals'] = result['main flow']
    result['CAS'] = result['CAS-Nr._y']
    # result.rename({'process':'Process', 'main flow': 'Included chemicals', 'CAS-Nr.':'CAS'}, axis=1)
    result['Included in'] = 'LAYER 3'
    result = result.loc[:, ['Process', 'Included chemicals', 'Included in', 'CAS']]
    print(result)
    print(result.columns)
    return result

# def write_frame_into_excelsheet(filename: Path, sheetname: str, dataframe: pd.DataFrame) -> None:
#     with pd.ExcelWriter(filename, engine='openpyxl', mode='a', data_only=True) as writer: 
#         workBook = writer.book
#         try:
#             workBook.remove(workBook[sheetname])
#         except:
#             print("Worksheet does not exist")
#         finally:
#             dataframe.to_excel(writer, sheet_name=sheetname,index=None)
#
def main(path: Path | str, config: ConfigParser, chosen: str) -> None:

    included_chemicals_master = Path(config['CM_CHEMICALS']['included_chemicals_master_file'])
    meta_data_flows_master_file = Path(config['CM_CHEMICALS']['meta_data_flows_master_file'])

    match chosen: 
        case "1":
            df = (
                get_data_for_reference_sheet(INPUT_PATH, 'raw_material_id', included_chemicals_master, meta_data_flows_master_file)
                .to_clipboard(index=None, header=None)
            )
        case "2":
            reference_sheet = pd.read_excel(INPUT_PATH, sheet_name='reference')
            reference_sheet['process'] = reference_sheet.apply(lambda x: f"reaction of {x['raw material 1']} and {x['raw material 2']}", axis=1)
            # print(reference_sheet)
            # print(reference_sheet.columns)
            print(reference_sheet.loc[:,['process', 'main flow']])
            reference_sheet.loc[:,['process', 'main flow']].to_clipboard(index=None)
        case "3":
            df = (
                get_data_for_addition_to_meta_data_flows_master_file(INPUT_PATH, 'raw_material_id', included_chemicals_master, meta_data_flows_master_file)
                .to_clipboard(index=None, header=None)
            )
        case "4":
            df = (
                get_data_for_addition_to_included_chemicals_file(INPUT_PATH, included_chemicals_master, meta_data_flows_master_file)
                .to_clipboard(index=None, header=None)
            )
        case _:
            print('WRONG INPUT!')
   
#print(pd.read_excel(meta_data_flows_master_file).columns.values.tolist())
    # processes = generate_processes_list_from_reference_sheet_and_raw_material_id(path)
    # remove_water_from_processes(processes)
    # try: 
    #     matrix = generate_matrix_from_list_of_processes(processes)
    # except: 
    #     raise RuntimeError("There is something wrong with the matrix generation!")
    # Copy file to avoid data loss 
    # copyfile(INPUT_PATH, INPUT_PATH.with_stem("reaction_extension_layer_inputcopy"))
    # Add matrix to original input excel
    # write_frame_into_excelsheet(filename=INPUT_PATH, sheetname='matrix_table', dataframe=matrix)
    # matrix.to_excel(INPUT_PATH.with_stem("reaction_extension_layer_matrix"), sheet_name='matrix_table')


if __name__ == '__main__':
    # INPUT_PATH = Path("../xlsx/reaction_extension_layer.xlsx")
    INPUT_PATH = Path(sys.argv[1])
    config = read_in_config()
    #INPUT_PATH = Path('../xlsx/')
    print("Welcome Traveler!\n It is dangerous to go alone. Choose one of these: \n ")
    print('1: Get Data for raw_materials sheet.')
    print('2: Get Data for process_id sheet.')
    print('3: Get Data for Additon to meta data flows master file.')
    print('4: Get Data for Addition to Included Chemicals.')
    chosen = input()
    main(INPUT_PATH, config, chosen)
