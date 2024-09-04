function [progress,flow_name_missing_CAS]=work_done_CAS(Model)
% calculate how many flows where already assessed by words
flow_name_missing_CAS={[]};
work_to_do=0;
flow_amounts=size(Model.meta_data_flows(2:end,4),1);
counter=0;
for i=2:size(Model.meta_data_flows,1)
    if isempty(Model.meta_data_flows{i,4})
       work_to_do=work_to_do+1;
       counter=counter+1;
       flow_name_missing_CAS(counter)=Model.meta_data_flows(i,1);
    end
end
flow_name_missing_CAS=flow_name_missing_CAS';
progress=(flow_amounts-work_to_do)/flow_amounts;
