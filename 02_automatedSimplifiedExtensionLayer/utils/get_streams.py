# This is a conversion of the matlab functions for database generation

def get_streams(file: Path) -> Streams:




# function streams = get_streams(file)
# %% Function to extract the in/output sterams from the IHS ecxel file
# streams.name = [];
# streams.cost = [];
# streams.cost_unit = [];
# streams.amount = [];
# streams.amount_unit = [];
# streams.cost_per_kg = [];
# streams.class = [];
#
# %% main product
# string = file{2,1};
# % find main product name 
# start_main_name = strfind(string,'Product:')+9; %stard of name
# end_main_name = strfind((string(start_main_name:end)),',  ')+start_main_name; % end of name
# end_main_name = end_main_name(1) -2;
# main_name = string(start_main_name:end_main_name);
#
# % find main product price 
# if strfind(string,'Price:') % check if there is a price for the product 
#     start_main_cost = strfind(string,'Price:')+7; %stard of name
#     end_main_cost = strfind((string(start_main_cost:end)),',  ')+start_main_cost; % end of name
#     end_main_cost = end_main_cost(1) -2;
#     main_cost = string(start_main_cost:end_main_cost);
#     %split in value and unit 
#     div = strfind(main_cost,' ');
#     main_cost_unit  = main_cost(div+1:end);
#     main_cost =  main_cost(1:div-1);
# else
#     main_cost = nan;
#     main_cost_unit = '$/kg';
# end
#
# % main product amount 
# main_amount = 1; 
#
# % main product unit 
# main_amount_unit = file{6,4}(5:end);
# if ~strcmp(main_amount_unit,'TONNE')
#     disp(main_amount_unit)
# end
#
# % set to output
# streams(1).name = main_name;
# streams(1).cost = str2double(main_cost);
# streams(1).cost_unit = main_cost_unit;
# streams(1).amount = main_amount;
# streams(1).amount_unit = main_amount_unit;
# streams(1).class = 1;
#
#
# %% raw materials 
# % find start raw materials
# start_raw_materials = 0;
# for i = 1:size(file,1)
#     if strcmp(file{i,1},'RAW MATERIALS') 
#         start_raw_materials = i+1;
#         break;
#     end
# end
#
# if start_raw_materials > 0
#     for i = start_raw_materials:size(file,1)
#         if isnan(file{i,1}) 
#             end_raw_materials = i-1;
#             break;
#         end
#     end
#
#
#     for i = start_raw_materials:end_raw_materials
#         streams(end+1).name = file{i,1};
#         streams(end).cost = file{i,2};
#         streams(end).cost_unit = file{i,3};
#         streams(end).amount = -file{i,4};
#         streams(end).amount_unit = file{i,5};
#         streams(end).class = 1;
#         streams(end).cost_per_kg = -file{i,6}/100;
#     end
# end
# %% by products
# start_by_product = 0;
# for i = 1:size(file,1)
#     if strcmp(file{i,1},'BY-PRODUCT CREDITS') 
#         start_by_product = i+1;
#         break;
#     end
# end
# if start_by_product > 0
#     for i = start_by_product:size(file,1)
#         if isnan(file{i,1}) 
#             end_by_product = i-1;
#             break;
#         end
#     end
#
#     for i = start_by_product:end_by_product
#         streams(end+1).name = file{i,1};
#         streams(end).cost = file{i,2};
#         streams(end).cost_unit = file{i,3};
#         streams(end).amount = -file{i,4};
#         streams(end).amount_unit = file{i,5};
#         streams(end).class = 1;
#         streams(end).cost_per_kg = file{i,6}/100;
#     end
# end
#
# %% utilities 
# start_utilities = 0;
# for i = 1:size(file,1)
#     if strcmp(file{i,1},'UTILITIES') 
#         start_utilities = i+1;
#         break;
#     end
# end
#
# if start_utilities >0
#     for i = start_utilities:size(file,1)
#         if isnan(file{i,1}) 
#             end_utilities = i-1;
#             break;
#         end
#     end
#
#     for i = start_utilities:end_utilities
#         streams(end+1).name = file{i,1};
#         streams(end).cost = file{i,2};
#         streams(end).cost_unit = file{i,3};
#         streams(end).amount = -file{i,4};
#         streams(end).amount_unit = file{i,5};
#         streams(end).class = 2;
#         streams(end).cost_per_kg = -file{i,6}/100;
#     end
# end
# streams = convert_units(streams);
# end
