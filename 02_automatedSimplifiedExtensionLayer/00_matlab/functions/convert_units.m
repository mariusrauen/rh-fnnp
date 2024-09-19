function  streams = convert_units(streams)

%% convert from TONNE to KG
for  i= 1:length(streams)
    if strcmp(streams(i).amount_unit,'TONNE')
        streams(i).amount = streams(i).amount*1e3;
        streams(i).amount_unit = 'kg';
        streams(i).unit_type = 'Mass';
    elseif strcmp(streams(i).amount_unit,'G')
        streams(i).amount = streams(i).amount/1e3;
        streams(i).amount_unit = 'kg';
        streams(i).unit_type = 'Mass';
    elseif strcmp(streams(i).amount_unit,'M3')
        streams(i).amount_unit = 'Nm3';
        streams(i).unit_type = 'Volumen';
    elseif strcmp(streams(i).amount_unit,'MNM3')
        streams(i).amount_unit = 'Nm3';
        streams(i).amount = streams(i).amount*1e3;
        streams(i).unit_type = 'Volumen';
    elseif strcmp(streams(i).amount_unit,'NM3')
        streams(i).amount_unit = 'Nm3';
        streams(i).unit_type = 'Volumen';
    elseif strcmp(streams(i).amount_unit,'KWH')
        streams(i).amount = streams(i).amount* 3.6; % convert KWH to MJ
        streams(i).amount_unit = 'MJ';
        streams(i).unit_type = 'Energy';
    elseif strcmp(streams(i).amount_unit,'EA')
        streams(i).amount = streams(i).amount; % Unit is unknown donsent apear in IHS Documentation
        streams(i).amount_unit = 'pcs';
        streams(i).unit_type = 'Pieces';
    elseif strcmp(streams(i).amount_unit,'MMCAL')
        streams(i).amount = streams(i).amount * 4.184; % convert MMCAL to MJ
        streams(i).amount_unit = 'MJ';
        streams(i).unit_type = 'Energy';
    elseif isnan(streams(i).cost_unit)
    else
        disp(['Amount Unit unknowen:',streams(i).amount_unit])
        streams(i).unit_type = 'Unknowen';
    end
end

%% convert price of streams
for  i= 1:length(streams)
    if strcmp(streams(i).cost_unit,'�/KG')
        streams(i).cost_unit = '$/KG';
        streams(i).cost = streams(i).cost/100;
    elseif strcmp(streams(i).cost_unit,'�/G')
        streams(i).cost_unit = '$/KG';
        streams(i).cost = streams(i).cost/100*1000;
    elseif strcmp(streams(i).cost_unit,'$/kg')
        streams(i).cost_unit = '$/KG';
    elseif strcmp(streams(i).cost_unit,'�/EA')
        streams(i).cost_unit = '$/EA';
        streams(i).cost = streams(i).cost/100;
    elseif strcmp(streams(i).cost_unit,'�/TONNE')
        streams(i).cost_unit = '$/KG';
        streams(i).cost = streams(i).cost/1000/100;
    elseif strcmp(streams(i).cost_unit,'�/M3')
        streams(i).cost_unit = '$/NM3';
        streams(i).cost = streams(i).cost/100;
    elseif strcmp(streams(i).cost_unit,'�/NM3')
        streams(i).cost_unit = '$/NM3';
        streams(i).cost = streams(i).cost/100;
    elseif strcmp(streams(i).cost_unit,'�/KWH')
        streams(i).cost_unit = '$/MJ';
        streams(i).cost = streams(i).cost/3.6/100;
    elseif strcmp(streams(i).cost_unit,'�/MMCAL')
        streams(i).cost_unit = '$/MJ';
        streams(i).cost = streams(i).cost/4.184/100;
    elseif isnan(streams(i).cost_unit)
    else
        disp(['Cost Unit unknowen:',streams(i).cost_unit])
    end
end
%% set the main stream to 1
% get factor 
factor = 1 / streams(1).amount;
for i = 1:length(streams)
   streams(i).amount = streams(i).amount * factor;  
end
end