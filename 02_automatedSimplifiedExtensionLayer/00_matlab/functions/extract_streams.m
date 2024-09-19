function streams = extract_streams(file,class)
    for i = 1:size(file,1)
        streams(end+1).name = file{i,1};
        streams(end).cost = file{i,2};
        streams(end).cost_unit = file{i,3};
        streams(end).amount = file{i,4};
        streams(end).amount_unit = file{i,5};
        streams(end).class = class;
    end
end