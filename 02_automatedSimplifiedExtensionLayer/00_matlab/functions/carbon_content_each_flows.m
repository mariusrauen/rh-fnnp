function [carbon_contents, names_of_flows]=carbon_content_each_flows(Model,path_inputs)


%% make final unit process database including waste flows

load([path_inputs,'elements.mat']);
load([path_inputs,'M_e.mat']);
load([path_inputs,'raw.mat']);

[carbon_contents, names_of_flows] = get_carbon_content(Model,elements,M_e,raw);

end