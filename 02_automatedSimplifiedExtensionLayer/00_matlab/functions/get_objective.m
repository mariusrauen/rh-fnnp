function [Model_new] = get_objective(Model,category)

disp('Objective function is written to Model.');

Model.matrices.objective.mean_values = ...
    Model.matrices.Q.mean_values(category,:)*Model.matrices.B.mean_values;

Model.matrices.objective.name = ...
    Model.meta_data_impact_categories(category,1);

Model_new = Model;

end

