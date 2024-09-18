function [process_errors,impacts_processes,s,y] = ...
    check_model_Ecoinvent(Model)

%% get objective for optimization test

category = 3; % Climate Change CML

[ Model ] = get_objective( Model , category );


%% check each process
[process_errors,impacts_processes,s,y] = test_each_process(Model);

% avoided_burden_NG = Model.matrices.A.mean_values * s;
% avoided_burden_NG = avoided_burden_NG(end,:)*Model.matrices.objective.mean_values(96);
% 
% impacts_processes = [num2cell(avoided_burden_NG)',impacts_processes];


end