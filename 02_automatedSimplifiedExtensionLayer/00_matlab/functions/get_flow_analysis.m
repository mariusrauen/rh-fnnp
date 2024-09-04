function [flows_in_processes]=get_flow_analysis(Model)

%% Inputs
for i=2:size(Model.meta_data_flows,1)

flow_analysis_inputs(i,1) = Model.meta_data_flows(i,1);
flow_analysis_inputs(i,2) = Model.meta_data_flows(i,4);
flow_analysis_inputs(i,3) = Model.meta_data_flows(i,2);
flow_analysis_inputs(i,4) = Model.meta_data_flows(i,6);

vector=(Model.matrices.A.mean_values(i-1,1:end)<0)';
col_names=Model.meta_data_processes(1,2:end);
processes= col_names([vector]);

for j=1:length(processes)
    flow_analysis_inputs{i,4+j} = processes{j};
end

end

flow_analysis_inputs{1,1}='flow';
flow_analysis_inputs{1,2}='CAS';
flow_analysis_inputs{1,3}='type';
flow_analysis_inputs{1,4}='unit';
flow_analysis_inputs{1,5}='using processes';
%% Outputs
for i=2:size(Model.meta_data_flows,1)

flow_analysis_outputs(i,1) = Model.meta_data_flows(i,1);
flow_analysis_outputs(i,2) = Model.meta_data_flows(i,14);
flow_analysis_outputs(i,3) = Model.meta_data_flows(i,2);
flow_analysis_outputs(i,4) = Model.meta_data_flows(i,6);

vector=(Model.matrices.A.mean_values(i-1,1:end)>0)';
col_names=Model.meta_data_processes(1,2:end);
processes= col_names([vector]);

for j=1:length(processes)
    flow_analysis_outputs{i,4+j} = processes{j};
end

end

flow_analysis_outputs{1,1}='flow';
flow_analysis_outputs{1,2}='SMILES';
flow_analysis_outputs{1,3}='type';
flow_analysis_outputs{1,4}='unit';
flow_analysis_outputs{1,5}='using processes';

%% save
flows_in_processes.outputs=flow_analysis_outputs;
flows_in_processes.inputs=flow_analysis_inputs;