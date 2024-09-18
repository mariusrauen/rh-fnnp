clear
clc

addpath(genpath('E:\COMPASS Database Build-Up\ZZ_final_database_scripts\07_test_models'));

%% Folder with inputs

model_name = 'Model_world.mat';
% model_name = 'GERMANY.mat';
 
%% Load Model

disp('Model tested');
temp=load(model_name);
temp=struct2cell(temp(1));
Model_test=temp{1,1};

clear temp


%% Give impact category to optimize
category=727; % Climate Change

%% test model
Model_test.matrices.A.mean_values(:,[322 328 332 335 339]) = 0;

    % choose any objective (see above)
    [ Model_test ] = get_objective( Model_test , category );
     
    % testing slow flows 
    [ flow_errors, impact_results ] = test_allocated_slow_RMe(Model_test);
   
    % testing slow processes 
    [ process_errors,impact_results_process ] = test_optimization_slow(Model_test);
    
    
   