import numpy as np
from scipy.optimize import linprog

def test_each_process(Model):
    # Setup the inequality matrix and objective function
    A_ineq = -Model['matrices']['A']['mean_values']
    objective = Model['matrices']['objective']['mean_values']

    # Determine the dimensions of the supply matrix
    columns_supply = A_ineq.shape[1]
    row_supply = A_ineq.shape[0]

    columns_numbers = list(range(columns_supply, columns_supply + row_supply))
    products_supply = [flow[0] for flow in Model['meta_data_flows'][1:]]

    add_supply = -np.eye(row_supply)
    objective_supply = np.ones(row_supply) * 1e10

    A_ineq_supply = np.hstack((A_ineq, add_supply))
    objective_supply = np.concatenate((objective, objective_supply))

    b_ineq = np.zeros(A_ineq.shape[0])
    b_ineq_supply = np.zeros(A_ineq_supply.shape[0])

    ub = np.ones(A_ineq.shape[1]) * 1e10
    lb = np.zeros(A_ineq.shape[1])

    ub_supply = np.ones(A_ineq_supply.shape[1]) * 1e10
    lb_supply = np.zeros(A_ineq_supply.shape[1])

    # Optimization settings
    options = {'method': 'highs', 'tol': 1e-10, 'options': {'maxiter': 50000, 'disp': False}}

    counter = 0
    counter_1 = 0
    process_errors = []

    impact_results_process = []
    s = []
    y = []

    for i in range(len(lb)):
        counter_1 += 1
        lb[i] = 1
        lb_supply[i] = 1

        # Run optimization using SciPy's linprog
        result = linprog(c=objective, A_ub=A_ineq, b_ub=b_ineq, bounds=list(zip(lb, ub)), **options)

        lb[i] = 0

        # Extract optimization results
        if result.success:
            impact = result.fun
            scaling_vector = result.x

            impact_results_process.append([impact, Model['meta_data_processes'][0][i+1], Model['meta_data_processes'][3][i+1]])

            if len(scaling_vector) > 0:
                s.append(scaling_vector)
                y.append(-np.dot(A_ineq, scaling_vector))
        else:
            s.append(np.zeros(A_ineq.shape[1]))
            y.append(np.zeros(A_ineq.shape[0]))

            print(f"Process unsuccessful: {Model['meta_data_processes'][0][i+1]}")
            counter += 1
            process_errors.append([Model['meta_data_processes'][0][i+1]])

            # Second optimization with supply constraints
            result_supply = linprog(c=objective_supply, A_ub=A_ineq_supply, b_ub=b_ineq_supply, bounds=list(zip(lb_supply, ub_supply)), **options)

            if result_supply.success:
                scaling_vector_supply = result_supply.x
                used_supplies = np.where(scaling_vector_supply[columns_numbers] > 0)[0]

                flows_not_producable = [products_supply[j] for j in used_supplies]
                process_errors[-1].append(flows_not_producable)

                print(f"Due to flow: {flows_not_producable}")

            lb_supply[i] = 0

    if not process_errors:
        print("All processes usable")

    return process_errors, impact_results_process, np.array(s).T, np.array(y).T
