function [Model,progress,meta_data_to_update,flows_in_processes]=make_flow_revision(Model, missing_meta_data)

%% flow revision

[Model] = perform_flow_revision(Model,missing_meta_data);

% check progress for meta data
[progress.progress_flow_revision_CAS,progress.missing_CAS] = work_done_CAS(Model);
[progress.progress_flow_revision_formular, progress.missing_formular]=work_done_formular(Model);
[progress.progress_flow_revision_Hv, progress.missing_HV] = work_done_Hv(Model);
[progress.progress_flow_revision_Smiles, progress.missing_Smiles] = work_done_SMILES(Model);

% generated table for meta data completion
[meta_data_to_update] = get_meta_data_to_update(Model.meta_data_flows);

[flows_in_processes]=  get_flow_analysis(Model);


end



