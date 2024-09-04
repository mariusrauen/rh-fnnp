function Model = perform_flow_revision(Model, all_flows_meta_data)

% first: delete possible space at end of flow names in Model meta data and all_flows_meta_data
% otherwise, matches cannot be found. Assumption: space at end of flow name is a mistake
% and not a legitimate distinction between flows

for i=2:size(Model.meta_data_flows,1)
    if Model.meta_data_flows{i,1}(end) == ' ' 
        Model.meta_data_flows{i,1}(end) = [];
    end
end

for i=2:size(all_flows_meta_data,1)
    if all_flows_meta_data{i,1}(end) == ' ' 
        all_flows_meta_data(end) = [];
    end
end


string_list=Model.meta_data_flows(:,1);

Model.meta_data_flows{1,15} = 'molecular mass';

for i=2:size(Model.meta_data_flows,1)
    % find row of model flow in all flow meta data
   row_flow=find(strcmp...
    (all_flows_meta_data(:,1),string_list(i)));
    

    % check for empty HHV and add 0 as a dummy (workaround for bad meta
    % data)
    
%     if isempty(Model.meta_data_flows{i,9})
%         Model.meta_data_flows{i,9} = 0;
%     end



    if isempty(row_flow)% continue because flow is not in meta data table
        continue
    end
    
    for k = row_flow' % go through all found rows
    
        if isequal(Model.meta_data_flows(i,6),all_flows_meta_data(k,6))&&... % unit is equal
                isequal(Model.meta_data_flows(i,2),all_flows_meta_data(k,2)) % type is equal

            Model.meta_data_flows(i,:) = all_flows_meta_data(k,:); % overwrite all meta data
          
        end
    end

end
%%
end