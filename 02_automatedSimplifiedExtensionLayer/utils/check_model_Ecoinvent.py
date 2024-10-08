from get_objective import get_objective
from test_each_process import test_each_process


def check_model_Ecoinvent(Model):
    """
    Function to check the model and process Ecoinvent data.
    """
    # Step 1: Get objective for optimization test
    category = 3  # Climate Change CML
    Model = get_objective(Model, category)

    # Step 2: Check each process
    process_errors, impacts_processes, s, y = test_each_process(Model)

    # Uncomment this section if you need to calculate avoided_burden_NG
    # avoided_burden_NG = np.dot(Model['matrices']['A']['mean_values'], s)
    # avoided_burden_NG = avoided_burden_NG[-1, :] * Model['matrices']['objective']['mean_values'][95]
    # impacts_processes = [[avoided_burden_NG[i]] + impacts_processes[i] for i in range(len(impacts_processes))]

    return process_errors, impacts_processes, s, y
