function [list_layer_1] = get_list_layer_1(Model, database_scope)

%% simplyfy inputs

meta_f = Model.meta_data_flows;
meta_p = Model.meta_data_processes;
A = Model.matrices.A.mean_values;

rows_layer_1 = find(cell2mat(database_scope(2:end,7))==1);
main_flows =  database_scope(rows_layer_1+1,3);

%% find processes with site products

A_pos = A>0;

counter = 0;

for i = 1:size(meta_p,2)-1
    
    pos_flows = find(A_pos(:,i));
    
    for j = 1:length(pos_flows)
        
        counter = counter + 1;
        
        name(counter,1) = meta_p(1,i+1);
        flow(counter,1) = meta_f(pos_flows(j)+1,1);
        is_main_flow(counter,1) = any(strcmp(main_flows,flow(counter,1)));

    end
    
    clear j
    clear pos_flows
    
end

rows_main = find(is_main_flow);

list_layer_1 = [name(is_main_flow),flow(is_main_flow)];

disp('DONE');


end