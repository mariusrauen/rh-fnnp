function [needed_from_ecoinvent]=get_flows_needed_ecoinvent(Model)

[flows_in_processes] = get_flow_analysis(Model);

row_non_empty = find(~cellfun(@isempty,flows_in_processes.inputs(:,5)));

input_flows = flows_in_processes.inputs(row_non_empty,1);

main_flows = Model.meta_data_processes(4,2:end);

main_flows(find(cellfun(@(C)...
any(isempty(C(:))), main_flows)))=[];

main_flows = unique(main_flows);

for i = 1:length(input_flows)

    flow_in_main_flows = find(strcmp(input_flows(i),main_flows));
    
    if ~isempty(flow_in_main_flows)
        input_flows(i) = {[]};
    end
    
end

input_flows = input_flows(~cellfun(@isempty,input_flows));

%% all flows that are only consumed not produced (except wastes)
% idx_1=and(all(Model.matrices.A.mean_values<=0,2),or(cell2mat(Model.meta_data_flows(2:end,2))==2,cell2mat(Model.meta_data_flows(2:end,2))==1));
% needed_from_ecoinvent_1=Model.meta_data_flows(idx_1,1);
% 
% %% all utilities
% idx_2=cell2mat(Model.meta_data_flows(2:end,2))==2;
% needed_from_ecoinvent_2=Model.meta_data_flows(idx_2,1);
% 
% %% all flows that have less than 20% positive entries
% B = Model.matrices.A.mean_values;
% B(B<=0)=0; % only positive
% 
% C = Model.matrices.A.mean_values;
% C(C>=0)=0; % only negative
% 
% positive_entries=sum(B>0,2);
% negative_entries=sum(C<0,2);
% 
% idx_3=and(positive_entries<negative_entries*0.20,or(cell2mat(Model.meta_data_flows(2:end,2))==2,cell2mat(Model.meta_data_flows(2:end,2))==1));
% needed_from_ecoinvent_3=Model.meta_data_flows(idx_3,1);

needed_from_ecoinvent = input_flows;  %unique(vertcat(needed_from_ecoinvent_1,needed_from_ecoinvent_2,needed_from_ecoinvent_3));

for i = 1:size(needed_from_ecoinvent,1)
    
    row = find(strcmp(Model.meta_data_flows(:,1),needed_from_ecoinvent(i,1)));
    
    if isequal(size(row,1),2) || isempty(row)
    continue
    end
    
    needed_from_ecoinvent(i,2)=Model.meta_data_flows(row,2);
    needed_from_ecoinvent(i,3)=Model.meta_data_flows(row,6);
    
end


end