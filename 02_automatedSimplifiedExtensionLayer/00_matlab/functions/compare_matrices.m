addpath('E:\COMPASS Database Build-Up\ZZ_final_database_scripts\07_test_models');
clc
clear a b f;
%% check A matrices
a = find(Model_1.matrices.A.mean_values~=Model_2.matrices.A.mean_values);
if isempty(a)
    disp('A matrices equal. ');
else
    disp('A matrices do not equal. ');
end

    


%% check B matrices
b = find(Model_1.matrices.B.mean_values~=Model_2.matrices.B.mean_values);
if isempty(b)
    disp('B matrices equal. ');
else
    disp('B matrices do not equal. ');
end



%% check F matrices
f = find(Model_1.matrices.F.mean_values~=Model_2.matrices.F.mean_values);
if isempty(f)
    disp('F matrices equal.');
else
    disp('F matrices do not equal. ');
end


