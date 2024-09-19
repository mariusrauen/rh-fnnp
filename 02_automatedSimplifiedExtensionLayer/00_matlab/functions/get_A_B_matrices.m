function [ Amatrix, Bmatrix ] = get_A_B_matrices(path_inputs)

%% Amatrix
load([path_inputs,'\layer_3\Amatrix.mat']);

%% Bmatrix
load([path_inputs,'\layer_3\Bmatrix.mat']);

end