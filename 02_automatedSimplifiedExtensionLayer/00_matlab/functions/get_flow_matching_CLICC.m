function [flow_matching]=get_flow_matching_CLICC(path_own_scenario, flow_matching)

    % get availible data files in scenario folder
    av_files=extractfield(dir(path_own_scenario),...
        'name')';
    
    % get flow matching tables
    if ismember('flow_matching_CLICC.xlsx',av_files)
    
        [~ , flow_matching_extra.names_1] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'A8:A1000');
        [flow_matching_extra.cats_1 , ~] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'B8:B1000');
        [~,flow_matching_extra.units_1] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'C8:C1000');
        [~ , flow_matching_extra.names_2] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'D8:D1000');
        [flow_matching_extra.cats_2 , ~] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'E8:E1000');
        [~,flow_matching_extra.units_2] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'F8:F1000');
        [flow_matching_extra.conv_factors , ~] = xlsread(fullfile(path_own_scenario,'flow_matching_CLICC.xlsx'),1,'G8:G1000');

        if ~isempty(flow_matching_extra.names_1)
            flow_matching.names_1=vertcat(flow_matching.names_1,flow_matching_extra.names_1);
            flow_matching.cats_1=vertcat(flow_matching.cats_1,flow_matching_extra.cats_1);
            flow_matching.units_1=vertcat(flow_matching.units_1,flow_matching_extra.units_1);
            flow_matching.names_2=vertcat(flow_matching.names_2,flow_matching_extra.names_2);
            flow_matching.cats_2=vertcat(flow_matching.cats_2,flow_matching_extra.cats_2);
            flow_matching.units_2=vertcat(flow_matching.units_2,flow_matching_extra.units_2);
            flow_matching.conv_factors=vertcat(flow_matching.conv_factors,flow_matching_extra.conv_factors);
        end
    else
        display('Not extra flow matching was performed.');
    end

end