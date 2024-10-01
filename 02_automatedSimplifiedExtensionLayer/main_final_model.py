# This is a python convertion of main_final_model.py 
## %% *SCRIPT TO BUILT THE TECHNOLOGY DATA INPUTS*
# % Original Author: Raoul.Meys@carbon-minds.com
# % Edits and Responsibility: Laura.Stellner@carbon-minds.com



"""
%% Make sure you are in the correct folder where code is stored
clear 
clc 
"""
## IMPORT LIBRARIES
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from numpy import array
from dataclasses import dataclass, field
from scipy.io import loadmat

from utils.excel_interaction import read_from_excel_cell
from utils.get_streams import Stream, get_streams
from utils.get_costs import get_costs
from utils.get_info import get_info

# Predefine data class
# @dataclass
# class Model:
#     technical_flows: list[Flow]
#     A_distribution: np.matrix
#     A_mean: np.matrix
#     A_std_dev: np.matrix
#     factor_requirements:
#     F_distribution:
#     F_mean: np.array
#     F_std_dev: np.matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

"""
% Get current directory and generate file path
workingPath = pwd;
addpath(genpath(pwd));

pathFunctions = fullfile(workingPath,'..','00_matlab','functions');
addpath(genpath(pathFunctions));
"""
# Get current directory and generate file path
# TODO Change hard coded path to e.g. 
#working_path = Path().cwd()
#sys.path.append(str(working_path))
working_path = Path("/mnt/c/Users/Jonas/Carbon Minds GmbH/Business - Dokumente/09 cm_chemicals database code/00_DatabaseGeneration/02_techModels/04_simplifiedExtensionLayer")
# TODO IMPORT FUNCTIONS:
pathfunctions = Path().relative_to("../02_matlab/functions")
# add functions path and subdirectories to sys.path?
#sys.path.append(str(path_functions))


"""
%% User interface to specify the required ecoinvent version and the ecoinvent system model (needed for the right elementary flows in the technology data)
promt1 = "Ecoinvent version required? (e.g. 3.9.1_2022)\n";
ecoinventVersion = input(promt1,'s');
promt2 = "Ecoinvent system model required? (e.g. apos or cut-off)\n";
ecoinventSystemModel = input(promt2,'s');
"""
# User interface to specify the required ecoinvent version and the ecoinvent system model (needed for the right elementary flows in the technology data)
# Uncomment shortcuts in final version
# TODO ecoinventVersion = input("Ecoinvent version required? (e.g. 3.9.1_2022)\n")
ecoinventVersion = "3.9.1_2022"
# TODO ecoinventSystemModel = input("Ecoinvent system model required? (e.g. apos or cut-off)\n")
ecoinventSystemModel = "cut-off"



"""
%% Add paths
pathInputPreprocessing = fullfile(workingPath,'..','..','01_ecoinvent','01_preprocessing','Output',ecoinventVersion,ecoinventSystemModel);
pathGlobalInput = fullfile(workingPath,'..','..','00_inputData');
pathInput = fullfile(workingPath,'..','01_input');
pathOutput = fullfile(workingPath,'00_TechModel',ecoinventVersion,ecoinventSystemModel);
mkdir(pathOutput)
pathRevisedModels = fullfile(workingPath,'02_revisedChemicals');
%pathRevisedModels_noAllocation = fullfile(workingPath,'02b_revisedChemicals_noAllocation');
pathInputWaste = fullfile(workingPath,'..','01_input','wasteCode');
pathEcoMatchingTable = fullfile(workingPath,'..','..','01_ecoinvent','03_modelExtensionLayer','01_input');
matchingTableName = ['EcoMatch_',ecoinventVersion,'_',ecoinventSystemModel,'.xlsx'];
correspondanceFile = 'CorrespondanceEF.xlsx';
ModelName = 'Layer3Model';
PathLayer23 = fullfile(workingPath,'..','IncludedChemicals.xlsx');
PathInputsModel = fullfile(workingPath,'..','03_extensionLayer','00_TechModel',ecoinventVersion,ecoinventSystemModel,'Layer2Model_Layer3Testing.mat');
"""
## Add paths
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



"""
%% Find all files in input directory
disp('Setting up model structure...');
file_list = dir(fullfile(pathInput,'dummy_IHS\')); 
% delete void entries
file_list = file_list(~cell2mat({file_list.isdir}));
"""
## Find all files in input directory
# Get list of .xslx files in the dummy_IHS directory, "glob for pattern matching"
logging.info('Setting up model structure...')
file_list = [file for file in (pathInput/'dummy_IHS').glob('*.xlsx') if file.is_file()]
#file_list = [file for file in (pathInput/'dummy_IHS').glob('*.xlsx')]



# JONAS --> MARIUS
"""
%% Remove hidden fields 
index = [];
for i = 1:length(file_list)
    if ~strcmp(file_list(i).name(1),'.')
        index(end+1) = i;
    end
end
file_list = file_list(index);
"""
# Remove hidden fields
file_list = [file for file in file_list if not file.name.startswith('.')]
# Output the list of files for verification
for file in file_list:
    logging.info(f"Found file: {file}")



"""
%% main loop to include excel files
for i = 1:length(file_list)
    % read file 
    [~,~,file] = xlsread(fullfile(pathInput,'dummy_IHS\',file_list(i).name));
    
    % get process
    process_list(i).streams = get_streams(file);
    process_list(i).costs = get_costs(file);
    process_list(i).info = get_info(file);   
end
"""
# Prepatory work from Jonas
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
process_list = []
for file in file_list:
    print(get_streams(file))

    data = pd.read_excel(file)

    streams = get_streams(data)  
    costs = get_costs(data)      
    info = get_info(data)

    process_list.append({'streams':streams, 'costs': costs, 'info': info})



'''
%% write results to COMPASS input
Model = write2model(process_list);
'''
# Write results to model, TODO: place actual function for write2model()
Model = write2model(process_list)



'''
%% Add Q and empty B matrix 
Model = add_Q_B(Model,pathInputPreprocessing);
'''
# Add Q and empty B matrix, TODO: place actual function for add_Q_B()
Model = add_Q_B(Model, path_input_preprocessing)



'''
%% set up final model
[Model] = clean_up_TM_symbols(Model);
Model.meta_data_processes{10,1} = 'allocation type';
% Include finished datasets
'''
# Set up the final model, TODO place actual function for clean_up_TM_symbols()
Model = clean_up_TM_symbols(Model)
Model['meta_data_processes'][9][0] = 'allocation type'   # Example: setting 'allocation type'



'''
%%
disp('Revised datasets are included...')
[ add_manual_processes,adding_files  ] = get_process_adding(pathRevisedModels);
'''
# TODO place actual function get_process_adding()
logging.info('Revised datasets are included...')
add_manual_processes, adding_files = get_process_adding(pathRevisedModels)



'''
deleted_EF = [];
matched_EF = [];
for i = 1:length(add_manual_processes)
    disp(adding_files(i+2));
    [ Model,elementary_flows_to_delete, elementary_flow_matching] = make_process_adding_V2(Model, add_manual_processes(i), correspondanceFile, pathGlobalInput , ecoinventVersion);
    deleted_EF = [deleted_EF;elementary_flows_to_delete];
    matched_EF = [matched_EF;elementary_flow_matching];
end
	% create unique values
    [~,idx]=unique(cell2mat(deleted_EF),'rows');
    deleted_EF = deleted_EF(idx,:);
	% create unique values
    matched_EF = cell2table(matched_EF);
    if ~isempty(matched_EF)
        matched_EF.Properties.VariableNames = {'name_new','compartment_new','subcompartment_new','name_old','compartment_old','subcompartment_old'};
        matched_EF = unique(matched_EF);
    end
%%    
'''
deleted_EF = []
matched_EF = []
for i, process in enumerate(add_manual_processes):
    logging.info(f'Processing file: {adding_files[i+2]}')
    # Ensure you have the equivalent function in Python for make_process_adding_V2
    Model, elementary_flows_to_delete, elementary_flow_matching = make_process_adding_V2(
        Model, process, correspondanceFile, pathGlobalInput, ecoinventVersion
    )
    deleted_EF.extend(elementary_flows_to_delete)
    matched_EF.extend(elementary_flow_matching)
import pandas as pd
# Convert to DataFrame to remove duplicates
deleted_EF_df = pd.DataFrame(deleted_EF).drop_duplicates()
# Convert matched_EF to DataFrame with specific column names
matched_EF_df = pd.DataFrame(matched_EF, columns=['name_new', 'compartment_new', 'subcompartment_new', 'name_old', 'compartment_old', 'subcompartment_old'])
# Remove duplicate rows in matched_EF_df
matched_EF_df = matched_EF_df.drop_duplicates()



'''
% Exclude IHS Dummy
Model.matrices.A.mean_values(:,1) = [];
Model.matrices.A.mean_values(1:4,:) = [];
Model.matrices.F.mean_values(:,1) = [];
Model.matrices.B.mean_values(:,1) = [];
Model.meta_data_processes(:,2) = [];
Model.meta_data_flows(2:5,:) = [];
disp('Excel files included')
'''
# Exclude the first column in A.mean_values, F.mean_values, and B.mean_values
Model['matrices']['A']['mean_values'] = np.delete(Model['matrices']['A']['mean_values'], 0, axis=1)
Model['matrices']['F']['mean_values'] = np.delete(Model['matrices']['F']['mean_values'], 0, axis=1)
Model['matrices']['B']['mean_values'] = np.delete(Model['matrices']['B']['mean_values'], 0, axis=1)
# Exclude the first four rows in A.mean_values
Model['matrices']['A']['mean_values'] = np.delete(Model['matrices']['A']['mean_values'], [0, 1, 2, 3], axis=0)
# Exclude the second column in meta_data_processes
Model['meta_data_processes'] = Model['meta_data_processes'].drop(Model['meta_data_processes'].columns[1], axis=1)
# Exclude rows 2 to 5 in meta_data_flows
Model['meta_data_flows'] = Model['meta_data_flows'].drop(Model['meta_data_flows'].index[1:5])
# Log that Excel files are included
logging.info('Excel files included')



'''
%% Include waste flows
% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);
disp('include waste flows');
[ Model ] = make_waste_and_elementary_flows(Model,pathInputWaste);
% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);
'''
# Include waste flows
# Get meta data
missing_meta_data = get_missing_meta_data(pathGlobalInput)
# Revise the model using the retrieved metadata
Model, _, meta_data_to_update, flows_in_processes = make_flow_revision(Model, missing_meta_data)
logging.info('Include waste flows')
# Include waste and elementary flows in the model
Model = make_waste_and_elementary_flows(Model, pathInputWaste)
# Get meta data again after including waste and elementary flows
missing_meta_data = get_missing_meta_data(pathGlobalInput)
# Revise the model again
Model, _, meta_data_to_update, flows_in_processes = make_flow_revision(Model, missing_meta_data)



'''
%% include process descriptions
disp('Process description is generated automatically.');
[Model] = generic_process_decription_Layer3(Model);
'''
# Include process descriptions
logging.info('Process description is generated automatically.')
# Generate and add process descriptions to the model
Model = generic_process_description_layer3(Model)



'''
%% load finalized system expansion
disp('Matrices are allocated ...');
A = Model.matrices.A.mean_values;
B = Model.matrices.B.mean_values;
F = Model.matrices.F.mean_values;
meta_data_processes = Model.meta_data_processes;
ProcessSize = size(meta_data_processes,1);
meta_data_flows = Model.meta_data_flows(:,[1,2,6]);
meta_data_elementary_flows = Model.meta_data_elementary_flows;
meta_data_monetary_flows = Model.meta_data_factor_requirements;
FlowSize = size(meta_data_monetary_flows,2);
meta_data_monetary_flows = reshape(meta_data_monetary_flows,[],FlowSize);
'''
# Load finalized system expansion
logging.info('Matrices are allocated ...')
# Extracting matrix values from the Model
A = Model['matrices']['A']['mean_values']
B = Model['matrices']['B']['mean_values']
F = Model['matrices']['F']['mean_values']
# Extracting metadata from the Model
meta_data_processes = Model['meta_data_processes']
# Determine the size of the processes metadata
ProcessSize = len(meta_data_processes)  # Equivalent to size(meta_data_processes, 1)
# Extract specific columns from the meta_data_flows (1st, 2nd, and 6th columns)
meta_data_flows = Model['meta_data_flows'].iloc[:, [0, 1, 5]]  # Adjust for Python's 0-based index
# Extract other metadata from the Model
meta_data_elementary_flows = Model['meta_data_elementary_flows']
meta_data_monetary_flows = Model['meta_data_factor_requirements']
# Reshape meta_data_monetary_flows to match MATLAB's behavior
FlowSize = meta_data_monetary_flows.shape[1]
meta_data_monetary_flows = meta_data_monetary_flows.values.reshape(-1, FlowSize)



'''
% set positive values for allocation calculation of utilities to zero
UtilityFlows = or(cell2mat(meta_data_flows(2:end,2)) == 2,string(meta_data_flows(2:end,1)) == "carbon dioxide");
Utilities = A(UtilityFlows,:);
MetaUtilities = meta_data_flows(find([0;UtilityFlows]),:);
meta_data_flows(find([0;UtilityFlows]),:) = [];
% Include meta data of flows
disp('Meta data of flows is updated...')
'''
# Set positive values for allocation calculation of utilities to zero
# Create a boolean mask for utility flows in meta_data_flows
UtilityFlows = (
    (meta_data_flows.iloc[1:, 1].astype(float) == 2) |  # Equivalent of `meta_data_flows(2:end, 2) == 2`
    (meta_data_flows.iloc[1:, 0] == "carbon dioxide")   # Equivalent of `string(meta_data_flows(2:end, 1)) == "carbon dioxide"`
)
# Extract Utilities using the boolean mask; we use .iloc[1:] to skip the first row
Utilities = A[UtilityFlows.values, :]
# Create a boolean mask for the rows to be removed (mimicking find([0; UtilityFlows]) in MATLAB)
zero_prepended = pd.Series([False]).append(UtilityFlows).reset_index(drop=True)  # Prepends a False to the beginning
# Extract MetaUtilities using the new mask
MetaUtilities = meta_data_flows[zero_prepended]
# Remove the identified utilities from meta_data_flows
meta_data_flows = meta_data_flows[~zero_prepended]  # Keep rows where the mask is False
# Include meta data of flows
print('Meta data of flows is updated...')


'''
%% Before performing allocation, Test if all flows used in energy allocation have a LHV >0
[flows_missingLHV_energy,problematic_processes_energy,flows_missingLHV_price,problematic_processes_price] =  perform_check_before_allocation(Model);
'''
# TODO place actual function for perform_check_before_allocation()
# Call the function and store the results
flows_missingLHV_energy, problematic_processes_energy, flows_missingLHV_price, problematic_processes_price = perform_check_before_allocation(Model)



'''
%% save the systemExpansion before the allocation
writecell(Model.meta_data_processes,...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','Process_meta_data');
writecell([[Model.meta_data_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_flows(2:end,:),num2cell(Model.matrices.A.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY A');
writecell([[Model.meta_data_elementary_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_elementary_flows(2:end,:),num2cell(Model.matrices.B.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY B');
writecell([[Model.meta_data_factor_requirements(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_factor_requirements(2:end,:),num2cell(Model.matrices.F.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY F');
'''
# Define the Excel file path
output_file_path = pathOutput / 'SystemExpansion_beforeAllocation.xlsx'
# Write 'Process_meta_data' sheet to Excel
process_meta_data_df = pd.DataFrame(Model['meta_data_processes'])
process_meta_data_df.to_excel(output_file_path, sheet_name='Process_meta_data', index=False)
# Write 'SUMMARY A' sheet to Excel
summary_a_data = pd.concat([
    pd.DataFrame(Model['meta_data_flows'].iloc[[0], :].values.tolist() + Model['meta_data_processes'].iloc[0, 1:].values.tolist()),  # Combine first row
    pd.concat([Model['meta_data_flows'].iloc[1:], pd.DataFrame(Model['matrices']['A']['mean_values'])], axis=1)  # Combine remaining rows
], ignore_index=True)
summary_a_data.to_excel(output_file_path, sheet_name='SUMMARY A', index=False)
# Write 'SUMMARY B' sheet to Excel
summary_b_data = pd.concat([
    pd.DataFrame(Model['meta_data_elementary_flows'].iloc[[0], :].values.tolist() + Model['meta_data_processes'].iloc[0, 1:].values.tolist()),  # Combine first row
    pd.concat([Model['meta_data_elementary_flows'].iloc[1:], pd.DataFrame(Model['matrices']['B']['mean_values'])], axis=1)  # Combine remaining rows
], ignore_index=True)
summary_b_data.to_excel(output_file_path, sheet_name='SUMMARY B', index=False)
# Write 'SUMMARY F' sheet to Excel
summary_f_data = pd.concat([
    pd.DataFrame(Model['meta_data_factor_requirements'].iloc[[0], :].values.tolist() + Model['meta_data_processes'].iloc[0, 1:].values.tolist()),  # Combine first row
    pd.concat([Model['meta_data_factor_requirements'].iloc[1:], pd.DataFrame(Model['matrices']['F']['mean_values'])], axis=1)  # Combine remaining rows
], ignore_index=True)
summary_f_data.to_excel(output_file_path, sheet_name='SUMMARY F', index=False)



'''
%% Allocation
A_alloc_mass = A;
A_alloc_mass(A_alloc_mass<0) = 0;
[ A_alloc_energy , ~ ] = get_energy_matrix_A(Model);
[ Model,A_alloc_price , ~ ] = get_price_matrix_A(Model);
'''
# Allocation - Creating Mass Allocation Matrix
A_alloc_mass = np.copy(A)  # Create a copy of A to avoid modifying the original
A_alloc_mass[A_alloc_mass < 0] = 0  # Set all negative values to 0
# Placeholder for energy allocation matrix and price allocation matrix functions
# TODO place actual get_energy_matrix() and get_price_matrix()
A_alloc_energy, _ = get_energy_matrix_A(Model)  # Assume this function returns two values; ignoring the second with _
Model, A_alloc_price, _ = get_price_matrix_A(Model)  # Assume this function modifies Model and returns two more values



'''
allocationType = nan(size(Model.meta_data_processes(10,:)));
idx_strings = cellfun(@ischar,Model.meta_data_processes(10,:));
allocationType(~idx_strings) = cell2mat(Model.meta_data_processes(10,~idx_strings));
allocationType(idx_strings) = 0;
alloc_procedure = allocationType(2:end);
'''
# Initialize allocationType as an array of NaN values with the same length as the 10th row of meta_data_processes
allocationType = np.full(meta_data_processes.shape[1], np.nan)
# Identify which elements in the 10th row of meta_data_processes are strings
idx_strings = meta_data_processes.iloc[9, :].apply(lambda x: isinstance(x, str))
# Convert non-string elements to float and assign them to allocationType
allocationType[~idx_strings] = pd.to_numeric(meta_data_processes.iloc[9, ~idx_strings], errors='coerce')
# Set string elements in allocationType to 0
allocationType[idx_strings] = 0
# Extract elements from the second to the end for alloc_procedure
alloc_procedure = allocationType[1:]



'''
if ~isequal(size(alloc_procedure,2),size(A,2))
    
    disp('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP');
    return
end
'''
# Check if the number of elements in alloc_procedure matches the number of columns in A
if alloc_procedure.size != A.shape[1]:
    print('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP')
    # Exit the function or script
    return



'''
%% Add the allocation types for the allocation descriptions that are strings (e.g. PCR)
% add the string descriptions of allogaion for PCRs from meta_data_processes
nan_idx = find(~cellfun(@isnumeric, meta_data_processes(10,:)))';
nan_idx = nan_idx(2:end);
allocationTypesNew = table(Model.meta_data_processes(10,:)');
allocationTypesNew.Var1(nan_idx) = meta_data_processes(10,nan_idx)';
% Replace all 0, 1, 2, 3 values with the replacement string
for i=1:height(allocationTypesNew)
    if isequal(allocationTypesNew.Var1(i),{0})
        allocationTypesNew.Var1{i} = 'no allocation needed in this unit process';
    elseif isequal(allocationTypesNew.Var1(i),{1})
        allocationTypesNew.Var1{i} = 'allocation via lower heating value';
    elseif isequal(allocationTypesNew.Var1(i),{2})
        allocationTypesNew.Var1{i} = 'allocation via mass';
    elseif isequal(allocationTypesNew.Var1(i),{3})
        allocationTypesNew.Var1{i} = 'allocation via price';
    end
end
allocationTypesNew.Var1 = string(allocationTypesNew.Var1);
allocationTypesNew.Var2 = allocationTypesNew.Var1 + repmat(". In case of energy co-production, allocation via avoided burden is applied for the co-produced energy.",height(allocationTypesNew),1);
allocationTypesNew.Var2(1) = allocationTypesNew.Var1(1);
Model.meta_data_processes(10,:) = cellstr(allocationTypesNew.Var2');
'''
# Assuming meta_data_processes is a pandas DataFrame and Model is a dictionary containing 'meta_data_processes'
# Find indices where the allocation descriptions are strings (e.g., PCRs)
nan_idx = [idx for idx, value in enumerate(Model['meta_data_processes'].iloc[9, :]) if not isinstance(value, (int, float))]
# Exclude the first index from nan_idx (similar to `nan_idx(2:end)` in MATLAB)
nan_idx = nan_idx[1:]
# Create a DataFrame to represent 'allocationTypesNew' table
allocationTypesNew = pd.DataFrame(Model['meta_data_processes'].iloc[9, :]).reset_index(drop=True)
allocationTypesNew.columns = ['Var1']
# Update 'Var1' with corresponding allocation descriptions
for i in range(len(allocationTypesNew)):
    if allocationTypesNew.loc[i, 'Var1'] == 0:
        allocationTypesNew.loc[i, 'Var1'] = 'no allocation needed in this unit process'
    elif allocationTypesNew.loc[i, 'Var1'] == 1:
        allocationTypesNew.loc[i, 'Var1'] = 'allocation via lower heating value'
    elif allocationTypesNew.loc[i, 'Var1'] == 2:
        allocationTypesNew.loc[i, 'Var1'] = 'allocation via mass'
    elif allocationTypesNew.loc[i, 'Var1'] == 3:
        allocationTypesNew.loc[i, 'Var1'] = 'allocation via price'
# Convert 'Var1' to string type
allocationTypesNew['Var1'] = allocationTypesNew['Var1'].astype(str)
# Create 'Var2' column with additional text appended
allocationTypesNew['Var2'] = allocationTypesNew['Var1'] + ". In case of energy co-production, allocation via avoided burden is applied for the co-produced energy."
allocationTypesNew.loc[0, 'Var2'] = allocationTypesNew.loc[0, 'Var1']  # Ensure first entry is just 'Var1'
# Update 'Model.meta_data_processes'
Model['meta_data_processes'].iloc[9, :] = allocationTypesNew['Var2'].values



'''
energy = alloc_procedure==1;
mass = alloc_procedure==2;
prices = alloc_procedure==3;
Amatrixforallocationfactors = zeros(size(A));
Amatrixforallocationfactors(:,energy) = A_alloc_energy(:,energy);
Amatrixforallocationfactors(:,mass) = A_alloc_mass(:,mass);
Amatrixforallocationfactors(:,prices) = A_alloc_price(:,prices);
'''
# Assuming A, A_alloc_energy, A_alloc_mass, A_alloc_price are numpy arrays
# Determine the allocation types (energy, mass, prices) from alloc_procedure
energy = alloc_procedure == 1
mass = alloc_procedure == 2
prices = alloc_procedure == 3
# Initialize Amatrixforallocationfactors with zeros, having the same shape as A
Amatrixforallocationfactors = np.zeros_like(A)
# Fill in the respective columns based on allocation type (energy, mass, prices)
Amatrixforallocationfactors[:, energy] = A_alloc_energy[:, energy]
Amatrixforallocationfactors[:, mass] = A_alloc_mass[:, mass]
Amatrixforallocationfactors[:, prices] = A_alloc_price[:, prices]


'''
%% ALLOCATION
% Amatrixforallocationfactors = A_alloc_mass;
A(UtilityFlows,:) = [];
Amatrixforallocationfactors(UtilityFlows,:) = [];
output = perform_allocation(...
    A,...
    B,...
    F,...
    Utilities,...
    Amatrixforallocationfactors,...
    Model.meta_data_processes,...
    meta_data_flows);
if any(sum(output.A>0)>1)
    disp('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP');
    return
end
'''
# ALLOCATION
# Optionally, set Amatrixforallocationfactors to A_alloc_mass if required.
# Amatrixforallocationfactors = A_alloc_mass
# Remove utility flows from A and Amatrixforallocationfactors matrices
A = np.delete(A, UtilityFlows, axis=0)
Amatrixforallocationfactors = np.delete(Amatrixforallocationfactors, UtilityFlows, axis=0)
# Perform the allocation using the perform_allocation function
output = perform_allocation(
    A,
    B,
    F,
    Utilities,
    Amatrixforallocationfactors,
    Model['meta_data_processes'],
    meta_data_flows
)
# Check for multifunctionality issues after allocation
if np.any(np.sum(output['A'] > 0, axis=0) > 1):
    print('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP')
    # Here we could raise an exception to stop the script
    raise RuntimeError('Multifunctionality issue detected in allocation.')


'''
scaling_A = max(output.A);
A = [output.A./scaling_A;output.Utilities./scaling_A];
F = output.F./scaling_A;
B = output.B./scaling_A;
MetaDataProcesses = output.meta_data_processes;
MetaDataTechnicalFlows = [meta_data_flows;MetaUtilities];
MetaDataMonetaryFlows = meta_data_monetary_flows;
MetaDataElementaryFlows = meta_data_elementary_flows;
Model.matrices.A.mean_values = A;
Model.matrices.B.mean_values = B;
Model.matrices.F.mean_values = F;
Model.meta_data_processes = MetaDataProcesses;
Model.meta_data_factor_requirements = MetaDataMonetaryFlows;
Model.meta_data_elementary_flows = MetaDataElementaryFlows;
Model.meta_data_flows(:,[1,2,6]) = MetaDataTechnicalFlows;
'''
# Scaling and updating matrices
# Calculate the scaling factor for matrix A
scaling_A = np.max(output['A'])
# Scale matrices A, Utilities, F, and B
A = np.vstack((output['A'] / scaling_A, output['Utilities'] / scaling_A))
F = output['F'] / scaling_A
B = output['B'] / scaling_A
# Update the metadata structures
MetaDataProcesses = output['meta_data_processes']
MetaDataTechnicalFlows = pd.concat([meta_data_flows, MetaUtilities], ignore_index=True)
MetaDataMonetaryFlows = meta_data_monetary_flows
MetaDataElementaryFlows = meta_data_elementary_flows
# Update the Model dictionary with new matrices and metadata
Model['matrices']['A']['mean_values'] = A
Model['matrices']['B']['mean_values'] = B
Model['matrices']['F']['mean_values'] = F
Model['meta_data_processes'] = MetaDataProcesses
Model['meta_data_factor_requirements'] = MetaDataMonetaryFlows
Model['meta_data_elementary_flows'] = MetaDataElementaryFlows
# Update specific columns in meta_data_flows to include MetaUtilities
Model['meta_data_flows'].iloc[:, [0, 1, 5]] = MetaDataTechnicalFlows



'''
% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);
'''
# Get meta data
missing_meta_data = get_missing_meta_data(pathGlobalInput)
# Revise the model with the updated meta data
Model, _, meta_data_to_update, flows_in_processes = make_flow_revision(Model, missing_meta_data)



'''
%% Exclude unwanted by-products
opts = spreadsheetImportOptions("NumVariables", 2);
% Specify sheet and range
opts.Sheet = "LAYER3";
% Specify column names and types
opts.VariableNames = ["IncludedProcesses", "IncludedChemical"];
opts.VariableTypes = ["string", "string"];
% Specify variable properties
opts = setvaropts(opts, ["IncludedProcesses", "IncludedChemical"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["IncludedProcesses", "IncludedChemical"], "EmptyFieldRule", "auto");
IncludedChemicalsTable = readtable(PathLayer23, opts, "UseExcel", false);
Layer2Chemicals = IncludedChemicalsTable.IncludedChemical(2:end);
Layer2Processes = IncludedChemicalsTable.IncludedProcesses(2:end);
ProcessesMainFlow = Model.meta_data_processes(4,2:end);
ProcessesNames = Model.meta_data_processes(1,2:end);
IncludedProcesses = zeros(length(Layer2Chemicals),1);
'''
# Define the Excel file reading options using pandas
excel_opts = {
    'sheet_name': 'LAYER3',  # Specify the sheet name
    'usecols': [0, 1],       # Specify the columns to use
    'dtype': {'IncludedProcesses': str, 'IncludedChemical': str}  # Specify column types
}
# Read the Excel file into a DataFrame
IncludedChemicalsTable = pd.read_excel(PathLayer23, **excel_opts)
# Convert the columns to preserve whitespace and handle empty fields (similar to MATLAB's setvaropts)
IncludedChemicalsTable['IncludedProcesses'] = IncludedChemicalsTable['IncludedProcesses'].str.strip()
IncludedChemicalsTable['IncludedChemical'] = IncludedChemicalsTable['IncludedChemical'].str.strip()
# Extract the Layer2Chemicals and Layer2Processes, skipping the header row
Layer2Chemicals = IncludedChemicalsTable['IncludedChemical'].iloc[1:].values
Layer2Processes = IncludedChemicalsTable['IncludedProcesses'].iloc[1:].values
# Extract processes' main flows and names from the Model's metadata, skipping the first column
ProcessesMainFlow = Model['meta_data_processes'].iloc[3, 1:].values
ProcessesNames = Model['meta_data_processes'].iloc[0, 1:].values
# Initialize IncludedProcesses array with zeros
IncludedProcesses = [0] * len(Layer2Chemicals)



'''
missing_Processes = [];
missing_MainFlows = [];
for i = 1:length(Layer2Chemicals)
    try
        IncludedProcesses(i,1) = find(strcmp(ProcessesMainFlow,Layer2Chemicals(i))&...
            strcmp(ProcessesNames,Layer2Processes(i)));
    catch
        missing_Processes = [missing_Processes;Layer2Processes(i)];
        missing_MainFlows = [missing_MainFlows;Layer2Chemicals(i)];
    end
end
'''
# Initialize lists to store missing processes and main flows
missing_Processes = []
missing_MainFlows = []
# Iterate over the Layer2Chemicals and Layer2Processes to find matches
for i in range(len(Layer2Chemicals)):
    try:
        # Find the index where both conditions are true
        match_index = next(idx for idx, (main_flow, name) in enumerate(zip(ProcessesMainFlow, ProcessesNames))
                           if main_flow == Layer2Chemicals[i] and name == Layer2Processes[i])
        IncludedProcesses[i] = match_index + 1  # MATLAB indices are 1-based; Python is 0-based
    except StopIteration:
        # If no match is found, add to missing lists
        missing_Processes.append(Layer2Processes[i])
        missing_MainFlows.append(Layer2Chemicals[i])



'''
if ~isempty(missing_Processes)
    for i=1:size(missing_Processes,1)
        disp(["ERROR: The process "+missing_Processes(i,1)+" with mainFlow "+missing_MainFlows(i,1)+" is in the IncludedChemicals table but is not included in the Model!"]);
    end
    error("ERROR: Please correct error in IncludedChemicals table for Layer2 or Model and run the code again!")
end
'''
# Check if there are any missing processes
if missing_Processes:
    # Iterate over the missing processes and display an error message for each
    for i in range(len(missing_Processes)):
        print(f"ERROR: The process {missing_Processes[i]} with mainFlow {missing_MainFlows[i]} is in the IncludedChemicals table but is not included in the Model!")

    # Raise an error to stop the execution and indicate the need for correction
    raise ValueError("ERROR: Please correct the error in IncludedChemicals table for Layer2 or Model and run the code again!")



'''
IncludedProcesses = sort(IncludedProcesses);
ExcludedProcesses = 1:size(Model.matrices.A.mean_values,2);
ExcludedProcesses(IncludedProcesses) = [];
ExcludedProcesses = sort(ExcludedProcesses);
ProcessMeta_excluded = Model.meta_data_processes;
ProcessMeta_excluded = ProcessMeta_excluded(:,[1, ExcludedProcesses+1]);
ProcessMeta_excluded = ProcessMeta_excluded([1,4],2:end)';
'''
# Sort IncludedProcesses
IncludedProcesses = np.sort(IncludedProcesses)
# Determine ExcludedProcesses
ExcludedProcesses = np.setdiff1d(np.arange(Model['matrices']['A']['mean_values'].shape[1]), IncludedProcesses)
# Sort ExcludedProcesses
ExcludedProcesses = np.sort(ExcludedProcesses)
# Extract and manipulate ProcessMeta_excluded
ProcessMeta_excluded = Model['meta_data_processes'].copy()
# Select columns for excluded processes and the first column (index 0 in Python)
ProcessMeta_excluded = ProcessMeta_excluded[:, np.hstack(([0], ExcludedProcesses + 1))]
# Select rows 1 and 4 (index 0-based in Python, so rows 0 and 3), and all columns from the 2nd onward
ProcessMeta_excluded = ProcessMeta_excluded[[0, 3], 1:].T  # Transpose to match the MATLAB output



'''
% Revise model
% A matrix and meta data
A = Model.matrices.A.mean_values(:,IncludedProcesses);
TechFlowMeta = Model.meta_data_flows;
TechFlows2Delete = any(A');
A(~TechFlows2Delete,:) = [];
TechFlowMeta = TechFlowMeta([1,find(TechFlows2Delete)+1],:);
'''
# Revise model
# Extract A matrix and meta data for IncludedProcesses
A = Model['matrices']['A']['mean_values'][:, IncludedProcesses]
# Copy the technical flow metadata
TechFlowMeta = Model['meta_data_flows'].copy()
# Determine rows in A matrix to delete
TechFlows2Delete = np.any(A.T, axis=1)
# Filter out rows in A that are to be deleted
A = A[~TechFlows2Delete, :]
# Update TechFlowMeta by including the first row and rows corresponding to TechFlows2Delete
TechFlowMeta = TechFlowMeta[np.hstack(([0], np.where(TechFlows2Delete)[0] + 1)), :]



'''
ProcessMeta = Model.meta_data_processes;
ProcessMeta = ProcessMeta(:,[1; IncludedProcesses+1]);
% Other matrices
B = Model.matrices.B.mean_values(:,IncludedProcesses);
F = Model.matrices.F.mean_values(:,IncludedProcesses);
% rewrite model
Model.matrices.A.mean_values = A;
Model.matrices.B.mean_values = B;
Model.matrices.F.mean_values = F;
Model.meta_data_processes = ProcessMeta;
Model.meta_data_flows = TechFlowMeta;
'''
# Extract Process Meta Data and filter based on IncludedProcesses
ProcessMeta = Model['meta_data_processes'].copy()
# Select only the first column and the columns corresponding to IncludedProcesses
ProcessMeta = ProcessMeta.iloc[:, [0] + (IncludedProcesses + 1).tolist()]
# Filter other matrices (B and F) based on IncludedProcesses
B = Model['matrices']['B']['mean_values'][:, IncludedProcesses]
F = Model['matrices']['F']['mean_values'][:, IncludedProcesses]
# Rewrite model matrices and meta data
Model['matrices']['A']['mean_values'] = A
Model['matrices']['B']['mean_values'] = B
Model['matrices']['F']['mean_values'] = F
Model['meta_data_processes'] = ProcessMeta
Model['meta_data_flows'] = TechFlowMeta



'''
%% Testing
% Load Layer1 including ecoinvent and convert Layer 2 to process adding
load(PathInputsModel);
[AddLayer2InclEcoinvent] = TransferModelToProcessAdding(Model);
'''
# Load Layer1 including ecoinvent and convert Layer 2 to process adding
# Load the .mat file using scipy.io.loadmat()
mat_data = loadmat(PathInputsModel)
# Access the 'Model' data from the loaded .mat file
Model_Ecoinvent_Layer1_and_2 = mat_data['Model_Ecoinvent_Layer1_and_2']
# Convert Layer 2 to process adding using the assumed defined function
AddLayer2InclEcoinvent = TransferModelToProcessAdding(Model)



'''
% combine models
[ Model_Layer1_and_2_and_3 ] = make_process_adding_V2_Layer23(Model_Ecoinvent_Layer1_and_2, AddLayer2InclEcoinvent);
'''
# Combine models using the assumed defined function
# TODO place actual function make_process_adding_V2_Layer23()
Model_Layer1_and_2_and_3 = make_process_adding_V2_Layer23(Model_Ecoinvent_Layer1_and_2, AddLayer2InclEcoinvent)



'''
% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model_Layer1_and_2_and_3,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model_Layer1_and_2_and_3, missing_meta_data);
disp('Ecoinvent inclusion.')
regions = {'GLO', 'RER', 'RoW','DE'}';
allocation_ecoinvent = 1;
'''
# Get metadata
missing_meta_data = get_missing_meta_data(pathGlobalInput)
# Perform flow revision
Model_Layer1_and_2_and_3, _, meta_data_to_update, flows_in_processes = make_flow_revision(Model_Layer1_and_2_and_3, missing_meta_data)
print('Ecoinvent inclusion.')
# Define regions
regions = ['GLO', 'RER', 'RoW', 'DE']
allocation_ecoinvent = 1



'''
[ Model_Ecoinvent_Layer1_and_2_and_3 ] = ...
    make_ecoinvent_one_addition(Model_Layer1_and_2_and_3,...
    pathInputPreprocessing,...
    pathEcoMatchingTable,...
    regions,...
    allocation_ecoinvent,...
    matchingTableName);
'''
# Make ecoinvent one addition to the model
Model_Ecoinvent_Layer1_and_2_and_3 = make_ecoinvent_one_addition(
    Model_Layer1_and_2_and_3,   # The model to add ecoinvent data to
    pathInputPreprocessing,     # Path for input preprocessing
    pathEcoMatchingTable,       # Path for eco matching table
    regions,                    # Regions to consider
    allocation_ecoinvent,       # Allocation type
    matchingTableName           # Matching table name
)



'''
% Check Model
[process_errors,impacts_processes,s,y] = ...
    check_model_Ecoinvent(Model_Ecoinvent_Layer1_and_2_and_3);
'''
# Check the ecoinvent model
process_errors, impacts_processes, s, y = check_model_Ecoinvent(Model_Ecoinvent_Layer1_and_2_and_3)



'''
%% check model for main flows
main_flow_missing = 0;
main_flow_value_zero = 0;
main_flows = Model.meta_data_processes(4,2:end);
'''
# Check model for main flows
main_flow_missing = False
main_flow_value_zero = False
main_flows = Model['meta_data_processes'].iloc[3, 1:].values  # Index 3 corresponds to MATLAB's 4th row, and 1: corresponds to 2:end in MATLAB



'''
for i = 1:size(main_flows,2)
    column = i;
    process = i+1;
    positive_values = Model.matrices.A.mean_values(:,i) > 0;
    positive_flows = Model.meta_data_flows(find(positive_values)+1,1);
    check_mainflow = any(contains(positive_flows,main_flows(i)));
    value_mainflow = Model.matrices.A.mean_values(positive_values,i);
    if ~check_mainflow
        disp(Model.meta_data_processes(1,process));
        main_flow_missing = 1;
    end
    if any(value_mainflow<=0)
        disp(Model.meta_data_processes(1,process),' with flow ',Model.meta_data_processes(4,process));
        main_flow_value_zero = 1;
    end 
end
'''
for i in range(len(main_flows)):
    column = i
    process = i + 1
    # Identify positive values in the 'A' matrix for the given column
    positive_values = Model['matrices']['A']['mean_values'][:, column] > 0
    # Find positive flows based on the positive_values mask and extract the corresponding flows from meta_data_flows
    positive_flows = Model['meta_data_flows'].iloc[np.where(positive_values)[0] + 1, 0].values
    # Check if any positive flow matches the main flow
    check_mainflow = any(main_flows[i] in flow for flow in positive_flows)
    # Get the main flow values for those that are positive
    value_mainflow = Model['matrices']['A']['mean_values'][positive_values, column]
    # Check if the main flow is missing
    if not check_mainflow:
        print(Model['meta_data_processes'].iloc[0, process])  # 0th row for 1st row in MATLAB
        main_flow_missing = True
    # Check if any main flow value is <= 0
    if any(value_mainflow <= 0):
        print(f"{Model['meta_data_processes'].iloc[0, process]} with flow {Model['meta_data_processes'].iloc[3, process]}")  # 0-based indexing
        main_flow_value_zero = True



'''
if isempty(find(any(y<-1e-10)))
    disp('y looks fine');
end

if isempty(process_errors{1,1})
    disp('processes look fine');
end

if main_flow_missing
    disp('some main flows are missing');
else
    disp('main flows look fine');
end

if main_flow_value_zero
    disp('some main flows are equal/below zero');
else
    disp('main flow values look fine');
end
'''
# Check if there are no values in y less than -1e-10
if not np.any(y < -1e-10):
    print('y looks fine')
# Check if process_errors is empty
if len(process_errors[0]) == 0:
    print('processes look fine')
# Check if any main flows are missing
if main_flow_missing:
    print('some main flows are missing')
else:
    print('main flows look fine')
# Check if any main flow values are equal to or below zero
if main_flow_value_zero:
    print('some main flows are equal/below zero')
else:
    print('main flow values look fine')



'''
%% SAVE RESULTS AND CLEAR MATLAB
save(fullfile(pathOutput,[ModelName,'.mat']),'Model');
save(fullfile(pathOutput,[ModelName,'_CustomTesting.mat']),'Model_Ecoinvent_Layer1_and_2');
%%
writecell(ProcessMeta',...
    fullfile(pathOutput,'ProcessMeta.xlsx'));
writecell(Model.meta_data_processes,...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','Process_meta_data');
writecell([[Model.meta_data_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_flows(2:end,:),num2cell(Model.matrices.A.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY A');
writecell([[Model.meta_data_elementary_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_elementary_flows(2:end,:),num2cell(Model.matrices.B.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY B');
writecell([[Model.meta_data_factor_requirements(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_factor_requirements(2:end,:),num2cell(Model.matrices.F.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY F');
rmpath(genpath(pathFunctions));
'''
import scipy.io as sio
import pandas as pd
import os
# Paths for saving the outputs
model_save_path = pathOutput / f"{ModelName}.mat"
model_custom_save_path = pathOutput / f"{ModelName}_CustomTesting.mat"
process_meta_excel_path = pathOutput / 'ProcessMeta.xlsx'
system_expansion_excel_path = pathOutput / 'SystemExpansion.xlsx'
# Save the Model and Model_Ecoinvent_Layer1_and_2 to .mat files
sio.savemat(model_save_path, {'Model': Model})
sio.savemat(model_custom_save_path, {'Model_Ecoinvent_Layer1_and_2': Model_Ecoinvent_Layer1_and_2})
# Save ProcessMeta to Excel
ProcessMeta_df = pd.DataFrame(ProcessMeta.T)  # Transpose for correct orientation
ProcessMeta_df.to_excel(process_meta_excel_path, index=False, header=False)
# Save System Expansion details to Excel with different sheets
with pd.ExcelWriter(system_expansion_excel_path) as writer:
    # Process Meta Data
    pd.DataFrame(Model['meta_data_processes']).to_excel(writer, sheet_name='Process_meta_data', index=False, header=False)
    # Summary A
    summary_A_data = pd.concat([
        pd.DataFrame([Model['meta_data_flows'].iloc[0, :].tolist() + Model['meta_data_processes'].iloc[0, 1:].tolist()]),
        pd.concat([Model['meta_data_flows'].iloc[1:, :], pd.DataFrame(Model['matrices']['A']['mean_values'])], axis=1)
    ])
    summary_A_data.to_excel(writer, sheet_name='SUMMARY A', index=False, header=False)
    # Summary B
    summary_B_data = pd.concat([
        pd.DataFrame([Model['meta_data_elementary_flows'].iloc[0, :].tolist() + Model['meta_data_processes'].iloc[0, 1:].tolist()]),
        pd.concat([Model['meta_data_elementary_flows'].iloc[1:, :], pd.DataFrame(Model['matrices']['B']['mean_values'])], axis=1)
    ])
    summary_B_data.to_excel(writer, sheet_name='SUMMARY B', index=False, header=False)
    # Summary F
    summary_F_data = pd.concat([
        pd.DataFrame([Model['meta_data_factor_requirements'].iloc[0, :].tolist() + Model['meta_data_processes'].iloc[0, 1:].tolist()]),
        pd.concat([Model['meta_data_factor_requirements'].iloc[1:, :], pd.DataFrame(Model['matrices']['F']['mean_values'])], axis=1)
    ])
    summary_F_data.to_excel(writer, sheet_name='SUMMARY F', index=False, header=False)
# The MATLAB command rmpath(genpath(pathFunctions)) does not have a direct equivalent in Python.
# Ensure any temporary directories or paths added to sys.path or similar are cleaned up if needed.



'''
if or(~isempty(flows_missingLHV_energy),~isempty(problematic_processes_energy))
    
    writetable(problematic_processes_energy,fullfile(pathOutput,'problematic_processes_and_flows_energy.xlsx'),...
    'Sheet','problematic Processes');
    writetable(flows_missingLHV_energy,fullfile(pathOutput,'problematic_processes_and_flows_energy.xlsx'),...
    'Sheet','problematic Flows missing LHV');

    disp('!!!WARNING!!!: Allocation by energy does not perform correctly!')
    disp('There are flows that are missing the LHV value but which occur as outputs in a process that is allocated by energy!')
    disp('Please check flows_missingLHV and problematic processes to see which flows and processes cause this issue.')
    disp('There are 2 things you can do:')
    disp('1. Add the LHV value in G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\meta_data_flows.xlsx')
    disp('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')
end
'''
# Check if there are missing LHV or problematic processes
if not flows_missingLHV_energy.empty or not problematic_processes_energy.empty:
    # Define paths for saving problematic processes and flows
    problematic_energy_path = pathOutput / 'problematic_processes_and_flows_energy.xlsx'
    # Write problematic processes to Excel
    with pd.ExcelWriter(problematic_energy_path) as writer:
        problematic_processes_energy.to_excel(writer, sheet_name='problematic Processes', index=False)
        flows_missingLHV_energy.to_excel(writer, sheet_name='problematic Flows missing LHV', index=False)
    # Display warnings and instructions
    print('!!!WARNING!!!: Allocation by energy does not perform correctly!')
    print('There are flows that are missing the LHV value but which occur as outputs in a process that is allocated by energy!')
    print('Please check flows_missingLHV and problematic processes to see which flows and processes cause this issue.')
    print('There are 2 things you can do:')
    print('1. Add the LHV value in G:\\Geteilte Ablagen\\09a_NEW_cm_chemicals_data\\01_final_technology_datasets\\00b_alignments\\meta_data_flows.xlsx')
    print('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')



'''
if or(~isempty(flows_missingLHV_price),~isempty(problematic_processes_price))
    
    writetable(problematic_processes_price,fullfile(pathOutput,'problematic_processes_and_flows_price.xlsx'),...
    'Sheet','problematic Processes');
    writetable(flows_missingLHV_price,fullfile(pathOutput,'problematic_processes_and_flows_price.xlsx'),...
    'Sheet','problematic Flows missing LHV');

    disp('!!!WARNING!!!: Allocation by price does not perform correctly!')
    disp('There are flows that are missing the price value but which occur as outputs in a process that may be allocated by price!')
    disp('Please check flows_missingprice and problematic processes to see which flows and processes cause this issue.')
    disp('There are 2 things you can do:')
    disp('1. Add the LHV value in G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\meta_data_flows.xlsx')
    disp('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')
end
disp('DONE')
'''
# Check if there are missing price values or problematic processes
if not flows_missingLHV_price.empty or not problematic_processes_price.empty:
    # Define paths for saving problematic processes and flows
    problematic_price_path = pathOutput / 'problematic_processes_and_flows_price.xlsx'
    # Write problematic processes and flows to Excel
    with pd.ExcelWriter(problematic_price_path) as writer:
        problematic_processes_price.to_excel(writer, sheet_name='problematic Processes', index=False)
        flows_missingLHV_price.to_excel(writer, sheet_name='problematic Flows missing LHV', index=False)
    # Display warnings and instructions
    print('!!!WARNING!!!: Allocation by price does not perform correctly!')
    print('There are flows that are missing the price value but which occur as outputs in a process that may be allocated by price!')
    print('Please check flows_missingprice and problematic processes to see which flows and processes cause this issue.')
    print('There are 2 things you can do:')
    print('1. Add the LHV value in G:\\Geteilte Ablagen\\09a_NEW_cm_chemicals_data\\01_final_technology_datasets\\00b_alignments\\meta_data_flows.xlsx')
    print('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')
print('DONE')







