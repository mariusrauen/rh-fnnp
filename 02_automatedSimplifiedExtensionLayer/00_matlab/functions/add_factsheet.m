function totalpath = add_factsheet(column,totalpath,master,chemicals)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;

final_dict_small = 'G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\Factsheet_chemical_neu_Oskar_28042020_small.xlsx';
final_dict_medium = 'G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\Factsheet_chemical_neu_Oskar_28042020_medium.xlsx';
final_dict_big = 'G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\Factsheet_chemical_neu_Oskar_28042020_big.xlsx';
destination = totalpath;

if column <= 18
    source = final_dict_small;
    copyfile (source,destination);
    % write name of chemical
    xlswrite(destination,cellstr({chemicals}), 'Tabelle1', 'A3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'M3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'AG3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1',  'AS3');
elseif column <= 36
    source = final_dict_medium;
    copyfile (source,destination);
    % write name of chemical
    xlswrite(destination,cellstr({chemicals}), 'Tabelle1', 'A3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'M3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'AG3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1',  'BA3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1',  'BM3');
else
    source = final_dict_big;
    copyfile (source,destination);
    %write name of chemical
    xlswrite(destination,cellstr({chemicals}), 'Tabelle1', 'A3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'M3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1', 'AG3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1',  'BA3');
    xlswrite(destination, cellstr({chemicals}), 'Tabelle1',  'BU3');
end

lns = find(strcmp(chemicals, master(:,1)));
if isempty(lns)
    error('');
end
xlswrite(destination,  master(lns,2),'Tabelle1', 'B8');
xlswrite(destination,  master(lns,3),'Tabelle1', 'D8');
xlswrite(destination,  master(lns,4),'Tabelle1', 'F8');


end

