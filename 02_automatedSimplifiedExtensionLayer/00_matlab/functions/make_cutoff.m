function [Model,Cutoff,processes] = ...
    make_cutoff(Model)

[processes] = get_processes(Model);

cut_off_parameter_over_all = 0.00;
cut_off_parameter_each_flow = 0.00;

[Model,Cutoff] =...
    perform_cutoff(Model,processes,...
    cut_off_parameter_over_all,cut_off_parameter_each_flow); 

end