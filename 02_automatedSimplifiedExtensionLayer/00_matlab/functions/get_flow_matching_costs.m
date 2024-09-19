function [flow_matching] = get_flow_matching_costs(path_alignments)

    % get availible data files in scenario folder
    av_files=extractfield(dir(path_alignments),...
        'name')';
    
    % get flow matching tables
    if ismember('flow_matching_extra_costs.xlsx',av_files)
    
        [~ , flow_matching.names_1] = xlsread(fullfile(path_alignments,'flow_matching_extra_costs.xlsx'),1,'A8:A1000');
        [~ , flow_matching.units_1] = xlsread(fullfile(path_alignments,'flow_matching_extra_costs.xlsx'),1,'B8:B1000');
        [~ , flow_matching.names_2] = xlsread(fullfile(path_alignments,'flow_matching_extra_costs.xlsx'),1,'C8:C1000');
        [~ , flow_matching.units_2] = xlsread(fullfile(path_alignments,'flow_matching_extra_costs.xlsx'),1,'D8:D1000');
        [flow_matching.conv_factors , ~] = xlsread(fullfile(path_alignments,'flow_matching_extra_costs.xlsx'),1,'E8:E1000');
        
    else
        display('Not extra flow matching of COSTs was performed.');
    end

end