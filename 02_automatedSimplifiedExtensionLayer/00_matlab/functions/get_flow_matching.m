function [flow_matching] = get_flow_matching(path_alignments)

    % get availible data files in scenario folder
    
        [~ , flow_matching.names_1] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'A8:A1000');
        [flow_matching.cats_1 , ~] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'B8:B1000');
        [~,flow_matching.units_1] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'C8:C1000');
        [~ , flow_matching.names_2] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'D8:D1000');
        [flow_matching.cats_2 , ~] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'E8:E1000');
        [~,flow_matching.units_2] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'F8:F1000');
        [flow_matching.conv_factors , ~] = xlsread(fullfile(path_alignments,'flow_matching_extra.xlsx'),1,'G8:G1000');


end