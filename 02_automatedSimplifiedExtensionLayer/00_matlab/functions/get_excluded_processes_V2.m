function [processes_to_exclude]=get_excluded_processes_V2(processes, included_processes)

file = processes;
output = logical(zeros(size(file(:,1),1),1));

for i = 1:size(included_processes,1)

    vector_1 = strcmp(file(:,1),included_processes(i));
    
    output = output|vector_1;
end

% save
file(:,2) = num2cell(output);
file(1,2) = {'exclude'};

processes_to_exclude = file;