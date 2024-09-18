function [process_adding] = TransferModelToProcessAdding(Model)


process_adding.meta_data_processes = Model.meta_data_processes;

process_adding.A = Model.matrices.A;
process_adding.meta_data_flows = [Model.meta_data_flows(2:end,1),Model.meta_data_flows(2:end,2),Model.meta_data_flows(2:end,6)];

process_adding.B = Model.matrices.B;
process_adding.meta_data_elementary_flows = [Model.meta_data_elementary_flows(2:end,1),Model.meta_data_elementary_flows(2:end,2),Model.meta_data_elementary_flows(2:end,3)];

rows_not_zero =  find(sum(process_adding.B.mean_values~=0,2)~=0);
process_adding.B.mean_values = process_adding.B.mean_values(rows_not_zero,:);
process_adding.meta_data_elementary_flows = process_adding.meta_data_elementary_flows(rows_not_zero,:);


process_adding.F = Model.matrices.F;
process_adding.meta_data_factor_requirements = [Model.meta_data_factor_requirements(2:end,1),Model.meta_data_factor_requirements(2:end,2)];
