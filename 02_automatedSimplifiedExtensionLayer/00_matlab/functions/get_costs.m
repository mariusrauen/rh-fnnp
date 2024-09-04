function costs = get_costs(file)
%% Function to extract the cost information form the IHS excel file

costs.name = [];
costs.value = [];
costs.unit = [];

capacity = file{7,10};
%get cost unit
base_unit = file{6,4}(5:end);
% get conversion factor
if strcmp(base_unit,'TONNE')
    factor = 1e-3;
elseif strcmp(base_unit,'MNM3')
    factor = 1e-6;
else
    dips('Unit Unkown in Cost')
    factor = 1;
end
unit = '$';

%% invest costs 
% inside battary
costs(1).name = file{9,8};
costs(1).value = file{9,10} * 1e8/1e9/capacity;
costs(end).unit = unit;
% off sites
costs(2).name = file{10,8};
costs(2).value = file{10,10}* 1e8/1e9/capacity;
costs(end).unit = unit;

%% operating costs
index = [17,18,19,20,21,23,24,26,28];
for i = index
    costs(end+1).name = file{i,8};
    costs(end).value = file{i,10}/100;
    costs(end).unit = unit;
end
end