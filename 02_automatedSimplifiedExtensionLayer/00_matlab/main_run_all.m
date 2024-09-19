%% Working Instruction:
%  1. You need to go into the individual matlab scripts and follow the working
%  instruction there.
%      1.a. main_generate_initial_model.m
%      1.b. main_prepare_flows.m
%      1.c. main_generate_se_model.m
%  
%  2. Afterwards, you need to make sure that you see the correct folder at
%  the left hand side of Matlab. You should see the folder content of the following foder: 
%  \00_DatabaseGeneration\02_techModels\00_matlab
%
%  3. Afterwards, you can run the code main_run_all.m
%
%  For any questions, please contact Laura Stellner, Aline Kalousdian, or
%  other members from the Database Team.

%% NO MODIFICATIONS FROM HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Do not change anything from here onwards %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run('main_generate_initial_model.m');
run('main_prepare_flows.m');
run('main_generate_se_model.m');