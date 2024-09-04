

clc;
clear all;

load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\raw.mat')
load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\elements.mat')
load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\M_e.mat')

%Change directory for the batch you want to investigate
load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\07_Layer2_3_Chemicals\01a_Modelling_Layer2\00c_FinalizedBatches\SYNTHESIS GAS\Model.mat')

[carbon_contents, names_of_flows] = get_carbon_content(Model,elements,M_e,raw);

carbon_contents = carbon_contents';
names_of_flows = names_of_flows';

%results = [carbon_contents,  names_of_flows]