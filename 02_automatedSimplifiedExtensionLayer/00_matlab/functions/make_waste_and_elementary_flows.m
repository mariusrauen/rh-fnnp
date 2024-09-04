function [Model_waste]=make_waste_and_elementary_flows(Model,path_inputs)


%% make final unit process database including waste flows

load(fullfile(path_inputs,'elements.mat'));
load(fullfile(path_inputs,'M_e.mat'));
load(fullfile(path_inputs,'raw.mat'));

[Model_waste] = get_waste_and_elementary_flows_direct_emissions(Model,elements,M_e,raw);

end