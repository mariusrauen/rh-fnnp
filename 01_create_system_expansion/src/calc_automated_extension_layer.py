import pandas as pd
import numpy as np
import math
import xlsxwriter
import pyodbc
import pyarrow.parquet as pq
import pyarrow as pa
import scipy as sps
from scipy import sparse
from numpy import array
from scipy.sparse import coo_matrix
import os

def load_prepare_data(input_file_path: str) -> pd.DataFrame:
    """Loads the prepared chemical reactions excel document into a pandas Dataframe."""

    reactions = pd.read_excel(input_file_path, sheet_name=['matrix_table', 'raw_material_id','process_id'])
    matrix_table = pd.DataFrame.from_dict(reactions['matrix_table'])
    global raw_materials
    raw_materials = pd.DataFrame.from_dict(reactions['raw_material_id'])
    global processes
    processes = pd.DataFrame.from_dict(reactions['process_id'])
    matrix_table = pd.merge(matrix_table, raw_materials, on=['raw_material_id'], how='left')
    global index_raw_materials
    index_raw_materials = raw_materials[['raw_material_id', 'raw materials']].sort_values(by= ['raw_material_id'])
    global index_processes
    index_processes = processes[['process_id','process']].sort_values(by= ['process_id'])
    return matrix_table

def carbon_content_check(matrix_table: pd.DataFrame) -> None:
    '''Check if carbon content is correct and write it to a excel sheet.'''
    # The regular expression captures to groups: 
    # 1) uppercase C's with a at least one number f, C(\d+) OR
    # 2) uppercase C's with another uppercase letter following (C[A-Z]).
    #
    # These are captured by the str.extract method and return a dataframe with two columns representing both cases
    # In the first case, if the regex finds no match, no carbon is present or the second case is matched
    # Therefore we add both series together to get the final carbon content.
    carbon_content_tmp = matrix_table['chemical formular'].str.extract(r'C(\d+)|(C[A-Z])', expand=True)
    carbon_content_s1 = carbon_content_tmp[1].fillna("0").map(lambda x: 0 if x=="0" else 1)
    carbon_content_s2 = carbon_content_tmp[0].fillna(0).astype(int)+carbon_content_s1
    carbon_content_check = pd.concat([matrix_table, carbon_content_s2], axis=1)
    carbon_content_check['carbon_content'] = carbon_content_check.coefficient*carbon_content_s2.astype(int)
    carbon_content_check.to_excel('../xlsx/carbon_content.xlsx')
    carbon = carbon_content_check.groupby(by=['process_id']).sum(numeric_only=True)
    carbon = carbon.reset_index()
    carbon.to_excel('../xlsx/carbon.xlsx')
    carbon_problem= carbon[carbon['carbon_content']!=0]
    carbon_problem= pd.merge(processes, carbon_problem[['process_id', 'carbon_content']], on='process_id')
    if carbon_problem.empty:
        print("Carbon content check was  carried out successfully!")
    else:
        print(carbon_problem)
        raise RuntimeError("The carbon balance of the process does not add up.")

def calculate_mass(matrix_table: pd.DataFrame) -> pd.DataFrame:
    '''Calculate the mass.'''
    matrix_table['weighted_mol_mass'] = matrix_table.coefficient*matrix_table['molecular mass']
    matrix_mass = pd.merge(matrix_table, processes, on='process_id')
    df = matrix_mass[matrix_mass['raw materials']==matrix_mass['main flow']]
    matrix_mass = pd.merge(matrix_mass, df[['main flow','weighted_mol_mass']], on='main flow')
    matrix_mass['mass'] = matrix_mass['weighted_mol_mass_x']/matrix_mass['weighted_mol_mass_y']
    matrix_mass = matrix_mass[['raw_material_id', 'process_id', 'raw materials', 'mass']]
    return matrix_mass

def consider_yield_095(matrix_mass: pd.DataFrame) -> pd.DataFrame:
    '''Consider a yield of 0.95, thus higher the mass of all inputs by 1/0.95.'''
    matrix_mass['mass_rm'] = matrix_mass[matrix_mass['mass']<0].mass*(1/0.95) 
    matrix_mass['mass_p'] = matrix_mass[matrix_mass['mass']>0].mass
    matrix_mass  = matrix_mass.fillna(0)
    matrix_mass['mass'] = matrix_mass.mass_rm + matrix_mass.mass_p
    matrix_mass = matrix_mass[['raw_material_id', 'process_id', 'raw materials', 'mass']]
    return matrix_mass

def calculate_energy_demand_Gendorf(matrix_mass: pd.DataFrame, matrix_table: pd.DataFrame) -> pd.DataFrame:
    '''Add energy demand according to Gendorf approach. This is achieved by summing up all produced masses for a given
       process and multipliying it with -2.0 for the thermal energy and -1.2 for the electricity.'''
    matrix_products = matrix_mass[matrix_mass['mass']>0]
    matrix = matrix_products.loc[:,['process_id', 'mass']].groupby(by='process_id', as_index=False).sum(numeric_only=True)

    # Duplicating the matrix for the estimation of the electricity.
    matrixA = matrix.copy()
    matrixA['raw_material_id'] = 0
    matrixA = pd.merge(matrixA, matrix_table[['raw materials','raw_material_id','coefficient']], on='raw_material_id', how='left')
    matrixA['mass'] = matrixA.mass*matrixA.coefficient
    
    # Duplicating the matrix for the estimation of thermal energy.
    matrixB = matrix.copy()
    matrixB['raw_material_id'] = 1
    matrixB = pd.merge(matrixB, matrix_table[['raw materials','raw_material_id','coefficient']], on='raw_material_id', how='left')
    matrixB['mass'] = matrixB.mass*matrixB.coefficient
    
    # Combining both matrixes.
    matrix_demand = pd.concat([matrixA, matrixB], axis=0)
    matrix_demand = matrix_demand[['raw_material_id', 'process_id', 'raw materials', 'mass']]
    return matrix_demand

def create_preprocessing_mass_table(matrix_mass: pd.DataFrame, matrix_demand: pd.DataFrame) -> pd.DataFrame:
    '''Create preprocessing excel with mass.'''
    matrix_table_mass = pd.concat([matrix_mass, matrix_demand], axis=0)
    matrix_table_mass = matrix_table_mass[['raw_material_id', 'process_id', 'mass']]
    with pd.ExcelWriter("../xlsx/matrix_table_mass.xlsx", engine='xlsxwriter') as writer:
                    matrix_table_mass.to_excel(writer, sheet_name="matrix_table")
                    raw_materials.to_excel(writer, sheet_name="raw_material_id")
                    processes.to_excel(writer, sheet_name="process_id")
    print("Preprocessing matrix_table_mass is created!")
    return matrix_table_mass

def matrix_creation(df_name: pd.DataFrame, column_name: str, row_name: str, data_name: str, index_row: str, index_column: str) -> pd.DataFrame:
    '''Convert matrix-table to system expansion and matrix.'''
    pm = np.array(list(df_name[column_name]))
    tfm = np.array(list(df_name[row_name]))
    coefficient = np.array(list(df_name[data_name]))
    x = sparse.coo_matrix((coefficient, (tfm, pm)),
    shape=(
        df_name[row_name].max()+1,
        df_name[column_name].max()+1,
    ),
)
    global matrix_variable
    matrix_variable = np.array(x.toarray())
    global matrix_df_variable
    matrix_df_variable = pd.DataFrame(data=matrix_variable, index= list(index_raw_materials[index_row]), 
    columns=list(index_processes[index_column]))
    print(matrix_df_variable)
    return matrix_df_variable

def create_system_expansion(matrixA: pd.DataFrame, output_file_path: str) -> None: # TODO: Write line with database version
    '''Create final system expansion excel.'''
    matrixA.index.names=['name']
    raw_materials_x = raw_materials[['name', 'category', '[unit choice]']]
    matrixA = pd.merge(raw_materials_x, matrixA, on='name')
    process_meta_data = pd.DataFrame(index= ['abbrevtion','process description', 'mainflow', 'location (choice)', 
    'exact location','capacity', 'unit per year (choice)', 'comments', 'type'] , columns=list(index_processes['process']))
    process_meta_data.loc['mainflow',:] =  np.array(list(processes['main flow']))
    process_meta_data.loc['unit per year (choice)',:] =  "t/a"
    process_meta_data.loc['type',:] =  2
    process_meta_data.index.names=['name']
    process_meta_data = process_meta_data.reset_index()
    matrixB = pd.DataFrame(index=None, columns=['name','compartment','subcompartment', 'unit']+list(index_processes['process']))
    matrixF = pd.DataFrame(index= None, columns=['name', 'unit']+list( index_processes['process']))
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
                    process_meta_data.to_excel(writer, sheet_name="Process_meta_data", index=False)
                    matrixA.to_excel(writer, sheet_name="SUMMARY A", index=False)
                    matrixB.to_excel(writer, sheet_name="SUMMARY B", index=False)
                    matrixF.to_excel(writer, sheet_name="SUMMARY F", index=False)
    print("Systemexpansion table is successfully created!")
