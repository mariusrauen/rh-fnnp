import numpy as np

def get_objective(Model, category):
    """
    Function to write the objective function to the Model.
    """
    print('Objective function is written to Model.')

    # Calculate the mean values for the objective function
    Model['matrices']['objective']['mean_values'] = (
        np.dot(Model['matrices']['Q']['mean_values'][category, :], Model['matrices']['B']['mean_values'])
    )

    # Assign the name for the objective
    Model['matrices']['objective']['name'] = Model['meta_data_impact_categories'][category][0]

    return Model
