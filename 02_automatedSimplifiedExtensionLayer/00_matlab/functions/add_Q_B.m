function Model = add_Q_B(Model,PathEcoinvent)


% load Q matrix

load(fullfile(PathEcoinvent , 'Q.mat'));

A = Model.matrices.A.mean_values;

%% Add B

meta_data_elementary_flows = ...
    {'name' 'compartment' 'sub compartment' 'unit'};

elementary_flows =...
    horzcat(cellstr(Q.col_name(:,1:end)), cellstr(Q.units));

Model.meta_data_elementary_flows =...
    vertcat(meta_data_elementary_flows, elementary_flows);

Model.matrices.B.mean_values = ...
    zeros(size(elementary_flows,1),size(A,2));
    
%% Add Q

Model.meta_data_impact_categories = Q.row_name;

Model.matrices.Q.mean_values = Q.mean_values;

end