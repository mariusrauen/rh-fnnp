function [Model]=manipulate_matrixes(Model,B_eco_mean,A_eco_mean,F_eco_mean,eco_process_meta)

%% Manipulate matrices and include B_mean values

%% meta data processes

Model.meta_data_processes=horzcat(...
    Model.meta_data_processes,...
    eco_process_meta);


%% A matrix
   
Model.matrices.A.mean_values=horzcat(...
    Model.matrices.A.mean_values,A_eco_mean);  

% Model.matrices.A.std_dev=horzcat(...
%     Model.matrices.A.std_dev,zeros(size(A_eco_mean)));
% 
% Model.matrices.A.distr_type=horzcat(...
%     Model.matrices.A.distr_type,2*ones(size(A_eco_mean)));
% 
% Model.matrices.A.col_names=horzcat(...
%     Model.matrices.A.col_names,eco_process_meta(1,:));



%% B matrix

B_eco_mean_zero=zeros(size(Model.matrices.B.mean_values,1),size(B_eco_mean,2));
B_eco_mean_zero(1:size(B_eco_mean,1),:)=B_eco_mean;

Model.matrices.B.mean_values=horzcat(...
    Model.matrices.B.mean_values,B_eco_mean_zero); 

% Model.matrices.B.std_dev=horzcat(...
%     Model.matrices.B.std_dev,zeros(size(B_eco_mean_zero)));
% 
% Model.matrices.B.distr_type=horzcat(...
%     Model.matrices.B.distr_type,2*ones(size(B_eco_mean_zero)));
% 
% Model.matrices.B.col_names=horzcat(...
%     Model.matrices.B.col_names,eco_process_meta(1,:));

%% Modify F

F_high=horzcat(Model.matrices.F.mean_values,...
    zeros(size(Model.matrices.F.mean_values,1),...
          size(F_eco_mean,2)));
      
F_low=horzcat(...
    zeros(size(F_eco_mean,1),size(Model.matrices.F.mean_values,2)),...
    F_eco_mean);
    
Model.matrices.F.mean_values=vertcat(F_high,F_low);
   
% Model.matrices.F.std_dev=zeros(size(Model.matrices.F.mean_values));
% 
% Model.matrices.F.distr_type=2*ones(size(Model.matrices.F.mean_values));
% 
% Model.matrices.F.col_names=horzcat(...
%     Model.matrices.F.col_names,eco_process_meta(1,:));

% make purchase processes
purchase(1:size(F_eco_mean,2),1)={'purchase '};
row_name_add_F=strcat(purchase,eco_process_meta(4,:)');
row_name_add_F(:,2)={'$'};
row_name_add_F(:,3)={[]};

Model.meta_data_factor_requirements=vertcat(Model.meta_data_factor_requirements,row_name_add_F);
% Model.matrices.F.row_names=vertcat(...
%     Model.matrices.F.row_names,...
%     row_name_add_F);

%% modify k, c, y

Model.matrices.k.mean_values=ones(size(Model.matrices.F.mean_values,1),1);
% Model.matrices.c.mean_values=zeros(size(Model.matrices.F.mean_values,1),1);
% Model.matrices.y.mean_values=zeros(size(Model.matrices.A.mean_values,1),1);
% 
% Model.matrices.k.std_dev=zeros(size(Model.matrices.k.mean_values));
% Model.matrices.c.std_dev=zeros(size(Model.matrices.c.mean_values));
% Model.matrices.y.std_dev=zeros(size(Model.matrices.y.mean_values));
% 
% Model.matrices.k.distr_type=2*ones(size(Model.matrices.k.mean_values));
% Model.matrices.c.distr_type=2*ones(size(Model.matrices.c.mean_values));
% Model.matrices.y.distr_type=2*ones(size(Model.matrices.y.mean_values));

% Model.matrices.k.row_names=Model.matrices.F.row_names;
% Model.matrices.c.row_names=Model.matrices.F.row_names;
% Model.matrices.y.row_names=Model.matrices.A.row_names;

end
