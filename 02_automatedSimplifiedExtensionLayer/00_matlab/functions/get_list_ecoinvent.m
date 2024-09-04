function Ecoinvent_data = get_list_ecoinvent(path_inputs)

%% load ecoinvent table
    path = fullfile(path_inputs,'Ecoinvent_data.mat');
    load(path);

end