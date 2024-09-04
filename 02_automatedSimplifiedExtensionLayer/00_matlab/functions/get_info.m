function info = get_info(file)
%% Function to extract additional information form the IHS fiel

string = file{2,1};
%% Find main product
start_main_name = strfind(string,'Product:')+9; %stard of name
end_main_name = strfind((string(start_main_name:end)),',  ')+start_main_name; % end of name
end_main_name = end_main_name(1) -2;
mainflow = string(start_main_name:end_main_name);

%% find location
start_main_name = strfind(string,'Geography:')+11; %stard of name
end_main_name = strfind((string(start_main_name:end)),',  ')+start_main_name; % end of name
end_main_name = end_main_name(1) -2;
location = string(start_main_name:end_main_name);

%% find process discription

for i = 1:size(file,1)
  if strcmp('PROCESS DESCRIPTION',file{i,1})
      description = file{i+1,1};
  end
end

info.name = file{1,1};
info.abbrevation = file{1,1};
info.process_description = description;

info.mainflow = mainflow;
info.location = location;
info.exact_location = 'unknown';

info.capacity = file{7,10}*1000;
info.unit_per_year = 't/a';
end