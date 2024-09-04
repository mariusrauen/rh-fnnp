function [flow_errors,impact_results] = test_each_flow(Model)

lb = zeros(size(Model.matrices.A.mean_values,2),1);

A_eq = Model.matrices.A.mean_values;

b_eq = zeros(size(A_eq,1),1);

ub=ones(size(Model.matrices.A.mean_values,2),1)*1e10;

objective = Model.matrices.objective.mean_values;

options = optimoptions('linprog',...
            'Algorithm','dual-simplex',...
            'OptimalityTolerance',1e-10,...
            'MaxIterations',50000,...
            'Display','off');

counter = 0;
counter_1 = 0;
flow_errors = {[]};

for i = 1:size(A_eq,1)
    
    b_eq(i,1) = 1;
    counter = counter+1;

    [scaling_vector,impact,~]=...
        linprog(objective,...
                [],...
                [],...
                A_eq,...
                b_eq,...
                lb,...
                ub,...
                options);
    b_eq(i) = 0;
    
     impact_results{counter,1} = impact;       
     impact_results{counter,2} = Model.meta_data_flows(i+1,1);
     impact_results{counter,3} = Model.meta_data_flows(i+1,6);
     impact_results{counter,4} = Model.meta_data_flows(i+1,11);
     
     if isempty(scaling_vector)
%         disp('Flow unsuccesfull:');
%         disp(Model.meta_data_flows(i+1,1));
        counter_1 = counter_1 +1;
        flow_errors(counter_1,1) = Model.meta_data_flows(i+1,1);
        continue;
     end
    
     
        
end

end