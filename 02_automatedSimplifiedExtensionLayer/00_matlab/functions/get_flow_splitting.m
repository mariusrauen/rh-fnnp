function [ flow_splitting ] = get_flow_splitting(path_alignments)

% for flow splitting no default file exists

% get availible data files in scenario folder

[~, flow_splitting.actual.processes.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'A6:A1000');
[~, flow_splitting.actual.flows.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'B6:B1000');
[flow_splitting.actual.flows.cats, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'C6:C1000');
[~, flow_splitting.actual.flows.units] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'D6:D1000');

[~, flow_splitting.target.flows_1.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'E6:E1000');
[flow_splitting.target.flows_1.cats, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'F6:F1000');
[~, flow_splitting.target.flows_1.units] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'G6:G1000');
[flow_splitting.target.flows_1.x, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'H6:H1000');

[~, flow_splitting.target.flows_2.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'I6:I1000');
[flow_splitting.target.flows_2.cats, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'J6:J1000');
[~, flow_splitting.target.flows_2.units] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'K6:K1000');
[flow_splitting.target.flows_2.x, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'L6:L1000');

[~, flow_splitting.target.flows_3.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'M6:M1000');
[flow_splitting.target.flows_3.cats, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'N6:N1000');
[~, flow_splitting.target.flows_3.units] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'O6:O1000');
[flow_splitting.target.flows_3.x, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'P6:P1000');

[~, flow_splitting.target.flows_4.names] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'Q6:Q1000');
[flow_splitting.target.flows_4.cats, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'R6:R1000');
[~, flow_splitting.target.flows_4.units] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'S6:S1000');
[flow_splitting.target.flows_4.x, ~] = xlsread(fullfile(path_alignments,'Control_flow_splitting.xlsx'),1,'T6:T1000');


end