function [processes]=get_processes(Model) %% inputs

for i = 2:size(Model.meta_data_processes,2)
    
    % make output file
    file{i,1} = Model.meta_data_processes{1,i};
    file{i,2} = 0;
    file{i,3} = Model.meta_data_processes{4,i};
    
    % find all input flows (value in A < 0)
    vector_1 = Model.matrices.A.mean_values(1:end,i-1) <0;
    
    % find all input flows that are main products (type ==1)
    vector_2 = cell2mat(Model.meta_data_flows(2:end,2))==1;
    
    % get all flow names that are inputs and main flows
    vector=and(vector_1,vector_2);
    row_names=Model.meta_data_flows(2:end,1);
    flows = row_names([vector]);
    
    % write to outpus file
    for j = 1:length(flows)
        file{i,3+j} = flows{j};
    end
end

for i = 1:size(file,1)
    for j = 1:size(file,2)
        if isempty(file{i,j})
            file{i,j} =0;
        end
    end
end

file{1,1} = 'processes';
file{1,2} = 'exclude';
file{1,3} = 'main product';
file{1,4} = 'inputs';

processes=file;
