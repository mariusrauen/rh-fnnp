function [Model]=make_matching(Model,flow_matching,flow_splitting)

%% flow matching

% 1. match & splitt flows

[Model] = get_flow_matching_Model( Model , flow_matching );
[Model] = get_flow_splitting_Model( Model , flow_splitting );



end