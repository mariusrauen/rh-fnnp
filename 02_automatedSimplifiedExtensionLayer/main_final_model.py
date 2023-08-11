# This is a python convertion of main_final_model.py 
#
# %% *SCRIPT TO BUILT THE TECHNOLOGY DATA INPUTS*
# % Original Author: Raoul.Meys@carbon-minds.com
# % Edits and Responsibility: Laura.Stellner@carbon-minds.com
from pathlib import Path
from utils.excel_interaction import read_from_excel_cell
import logging
from utils.get_streams import Stream, get_streams
from pprint import pprint

# Define Path Variables
#working_path = Path().cwd()
working_path = Path("/mnt/c/Users/Jonas/Carbon Minds GmbH/Business - Dokumente/09 cm_chemicals database code/00_DatabaseGeneration/02_techModels/04_simplifiedExtensionLayer")
## TODO IMPORT FUNCTIONS : pathfunctions = Path().relative_to("../02_matlab/functions")

# User interface to specify the required ecoinvent version and the ecoinvent system model (needed for the right elementary flows in the technology data)
# Uncomment shortcuts in final version
#ecoinventVersion = input("Ecoinvent version required? (e.g. 3.9.1_2022)\n")
ecoinventVersion = "3.9.1_2022"
#ecoinventSystemModel = input("Ecoinvent system model required? (e.g. apos or cut-off)\n")
ecoinventSystemModel = "cut-off"
# Add more paths
pathInputPreprocessing = working_path/'..'/'..'/'01_ecoinvent'/'01_preprocessing'/'Output'/ecoinventVersion/ecoinventSystemModel
pathGlobalInput = working_path/'..'/'..'/'00_inputData'
pathInput = working_path/'..'/'01_input'
pathOutput = working_path/'00_TechModel'/ecoinventVersion/ecoinventSystemModel
pathOutput.mkdir(parents=True, exist_ok=True)
pathRevisedModels = working_path/'02_revisedChemicals'
# UNCOMMEND FOR NO ALLOCATION: pathRevisedModels_noAllocation = working_path/'02b_revisedChemicals_noAllocation'
pathInputWaste = working_path/'..'/'01_input'/'wasteCode'
pathEcoMatchingTable = working_path/'..'/'..'/'01_ecoinvent'/'03_modelExtensionLayer'/'01_input'

# Generate some variables
matchingTableName = f"EcoMatch_{ecoinventVersion}_{ecoinventSystemModel}.xlsx"
correspondanceFile = 'CorrespondanceEF.xlsx'
ModelName = 'Layer3Model'
PathLayer23 = working_path/'..'/'IncludedChemicals.xlsx'
PathInputsModel = working_path/'..'/'03_extensionLayer'/'00_TechModel'/ecoinventVersion/ecoinventSystemModel/'Layer2Model_Layer3Testing.mat'


## Find all files in input directory

logging.info('Setting up model structure...')
file_list = [file for file in (pathInput/'dummy_IHS').glob('*.xlsx')]

for file in file_list:
    pprint(get_streams(file))
# %% main loop to include excel files
#
# for i = 1:length(file_list)
#     % read file 
#     [~,~,file] = xlsread(fullfile(pathInput,'dummy_IHS\',file_list(i).name));
#[num,txt,raw] = xlsread('myExample.xlsx')
# num =
#      1     2     3
#      4     5   NaN
#      7     8     9
#
# txt = 
#     'First'    'Second'    'Third'
#     ''         ''          ''     
#     ''         ''          'x'    
#
# raw = 
#     'First'    'Second'    'Third'
#     [    1]    [     2]    [    3]
#     [    4]    [     5]    'x'    
#     [    7]    [     8]    [    9]
# #     
#     % get process
#     process_list(i).streams = get_streams(file);
#     process_list(i).costs = get_costs(file);
#     process_list(i).info = get_info(file);   
# end 
#
# %% write results to COMPASS input
#
# Model = write2model(process_list);
#
# %% Add Q and empty B matrix 
#
# Model = add_Q_B(Model,pathInputPreprocessing);
#
# %% set up final model
#
# [Model] = clean_up_TM_symbols(Model);
# Model.meta_data_processes{10,1} = 'allocation type';
# %
