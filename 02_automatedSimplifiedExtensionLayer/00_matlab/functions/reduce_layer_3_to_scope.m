function [Amatrix, Bmatrix] = ...
    reduce_layer_3_to_scope(Amatrix, Bmatrix, database_scope)

%% find layer 3 chemicals 
layers = cell2mat(database_scope(2:end,7));

layer_3_chemicals = database_scope((find(layers == 3)+1),4);

%% find processes of layer 3
counter = 0;
if isempty(layer_3_chemicals)
    [Amatrix] = [];
    [Bmatrix] = [];
    return
end

for i = 1:length(layer_3_chemicals)
    
    if isequal(size(find(strcmp(Amatrix.chemicals(:,1),layer_3_chemicals(i))),1),1)
    counter = counter + 1;
    column_layer_3(counter) = find(strcmp(Amatrix.chemicals(:,1),layer_3_chemicals(i)));
    end
    
end

%% reduce A

Amatrix.mean_values = Amatrix.mean_values(:,column_layer_3);

Amatrix.processes = Amatrix.processes(:,column_layer_3);

% delete zero rows
exclude_rows = [];
for i = 1 : size(Amatrix.mean_values,1)
    if Amatrix.mean_values(i,:) == 0
        exclude_rows(end+1,1) = i;
    end
end

% Modify Matrix A
Amatrix.mean_values(exclude_rows,:) = [];

% Modify meta data flows
Amatrix.chemicals(exclude_rows,:) = [];

%% reduce B

Bmatrix.mean_values = Bmatrix.mean_values(:,column_layer_3);

Bmatrix.processes = Bmatrix.processes(:,column_layer_3);

end