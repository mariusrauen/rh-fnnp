function [progress, flow_names_missing_Hv]=work_done_Hv(Model)
% calculate how many flows where already assessed by words

work_to_do=0;
flow_amounts=size(Model.meta_data_flows(2:end,4),1);
counter=0;
flow_names_missing_Hv={[]};
for i=2:size(Model.meta_data_flows,1)
    if isempty(Model.meta_data_flows{i,9})
       work_to_do=work_to_do+1;
       counter=counter+1;
       flow_names_missing_Hv(counter)=Model.meta_data_flows(i,1);
    end
end
flow_names_missing_Hv=flow_names_missing_Hv';
progress=(flow_amounts-work_to_do)/flow_amounts;
