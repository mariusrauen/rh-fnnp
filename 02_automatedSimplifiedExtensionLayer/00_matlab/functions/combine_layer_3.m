function [Model_layer_1_2_3, elementary_flows_not_found] = combine_layer_3(Amatrix, Bmatrix, Model,path_inputs);


%%
clear A_new B_new F_new processes_new flows_new

A = Model.matrices.A.mean_values;
B = Model.matrices.B.mean_values;
F = Model.matrices.F.mean_values;
processes = Model.meta_data_processes;
flows = Model.meta_data_flows;
elementary_flows = Model.meta_data_elementary_flows;

A_layer_3 = Amatrix.mean_values;
B_layer_3 = Bmatrix.mean_values;
processes_layer_3 = Amatrix.processes;
flows_layer_3 = Amatrix.chemicals;
elementary_flows_layer_3 = Bmatrix.chemicals;

%% make flow matching

[flow_matching] = make_flow_matching_by_CAS(flows_layer_3, flows, path_inputs);



%%

A_new = [A, zeros(size(A,1),size(A_layer_3,2));...
         zeros(size(A_layer_3,1),size(A,2)), A_layer_3];
     
%% B needs revision if layer 3 is included    

B_layer_3_size_B = zeros(size(B,1),size(B_layer_3,2));
counter = 0;
for i = 1 : size(B_layer_3,2) % processes in B_layer_3
    
    for j = 1 : size(B_layer_3,1) % flows in B_layer_3
        
        rows_elementary_B = find(strcmp(elementary_flows(:,1),elementary_flows_layer_3(j,1))); % matched flow names in B
        found = 0;
        
        if isempty(rows_elementary_B)
            counter = counter + 1;
            elementary_flows_not_found(counter,1) = elementary_flows_layer_3(j,1);
            elementary_flows_not_found(counter,2) = elementary_flows_layer_3(j,2);
            elementary_flows_not_found(counter,3) = elementary_flows_layer_3(j,3);
            continue
        end
                    
        for k = rows_elementary_B'
            
            if isequal(elementary_flows(k,2),elementary_flows_layer_3(j,2)) &&...
                    isequal(elementary_flows(k,3),elementary_flows_layer_3(j,3))
                
                found = 1;
                
                B_layer_3_size_B(k-1,i) = B_layer_3(j,i);
                
                if found
                    break                  
                end
            
            
            end
        
        
        end
    
    
    end

end

B_new = [B, B_layer_3_size_B];

%% 

F_new = [F, zeros(size(F,1),size(B_layer_3,2))];

processes_new = [processes, cell(size(processes,1),size(processes_layer_3,2))];

processes_new(1,(end-size(processes_layer_3,2)+1):end) = processes_layer_3;
processes_new(9,(end-size(processes_layer_3,2)+1):end) = {'layer 3'};
processes_new(10,(end-size(processes_layer_3,2)+1):end) = {3};

flows_new = [flows;cell(size(flows_layer_3,1),size(flows,2))];

flows_new((end-size(flows_layer_3)+1):end,1) = flows_layer_3(:,1);
flows_new((end-size(flows_layer_3)+1):end,4) = flows_layer_3(:,3);
flows_new((end-size(flows_layer_3)+1):end,14) = flows_layer_3(:,4);
flows_new((end-size(flows_layer_3)+1):end,6) = flows_layer_3(:,2);
flows_new((end-size(flows_layer_3)+1):end,2) = {1};

Model.matrices.A.mean_values = A_new;
Model.matrices.B.mean_values = B_new;
Model.matrices.F.mean_values = F_new;
Model.meta_data_processes = processes_new;
Model.meta_data_flows = flows_new;

%% match flows and overwrite model

[ Model ] = matching_flows_layer_1_2_3( Model , flow_matching );

Model_layer_1_2_3 = Model;

end