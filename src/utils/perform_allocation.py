import numpy as np

def perform_allocation(Atoallocate, B, F, Utilities, Amatrixforallocationfactors, meta_data_processes, meta_data_flows):
    # Analyze Matrices
    A = Amatrixforallocationfactors
    m, n = A.shape  # matrix size

    # Output flows
    IsOutFlow = Atoallocate > 0
    NumOutFlowsOfProcess = np.sum(IsOutFlow, axis=0)
    IsMultiOutProcess = NumOutFlowsOfProcess > 1
    IsSingleOrNonOutProcess = NumOutFlowsOfProcess <= 1
    NumSingleOrNonOutProcess = np.sum(IsSingleOrNonOutProcess)
    NumOutFlowsOfMultiOutProcess = np.sum(np.sum(IsOutFlow[:, IsMultiOutProcess], axis=0))

    # Vector to determine size of matrices
    n_beta = np.zeros(n)
    n_beta[IsSingleOrNonOutProcess] = 1
    n_beta[IsMultiOutProcess] = 1 + NumOutFlowsOfProcess[IsMultiOutProcess]

    # 'Only-output' Matrix, required for allocation
    A_out = np.zeros((m, n))
    A_out[IsOutFlow] = A[IsOutFlow]
    A_sumOut = np.sum(A_out, axis=0)

    # Construct T_0, T_1, and U matrices based on column generation
    Identity_n = np.eye(n)
    T_0 = np.zeros((n, int(np.sum(n_beta))))
    T_1 = np.zeros((n, int(np.sum(n_beta))))
    U = np.zeros((m, int(np.sum(n_beta))))

    c = 0  # column counter
    for i in range(n):  # for all processes
        for j in range(int(n_beta[i])):  # for all flows in each process
            T_1[:, c] = Identity_n[:, i]
            if NumOutFlowsOfProcess[i] <= 1:  # Single-functional or non-output process
                T_0[:, c] = Identity_n[:, i]
                U[:, c] = 1
            else:  # Multi-output process allocation
                if j == 0:  # Original multi-product process
                    T_0[:, c] = Identity_n[:, i]
                    U[:, c] = 1
                    U[IsOutFlow[:, i], c] = 0  # Zeros in functional flows rows
                else:  # Duplicated columns
                    T_0[:, c] = np.zeros(n)
                    IdxOutFlows = np.where(IsOutFlow[:, i])[0][:j]
                    U[IdxOutFlows[-1], c] = 1
            c += 1

    # Construct C matrix based on row generation
    C = np.zeros((int(np.sum(n_beta)), NumSingleOrNonOutProcess + NumOutFlowsOfMultiOutProcess))

    # Expanded meta-data array
    new_meta_data_processes = np.empty((meta_data_processes.shape[0], NumOutFlowsOfMultiOutProcess + NumSingleOrNonOutProcess), dtype=object)
    new_meta_data_processes[:, 0] = meta_data_processes[:, 0]

    r = 0  # row counter
    c = 0  # column counter
    for i in range(n):  # for all processes
        if NumOutFlowsOfProcess[i] <= 1:  # Single-functional or non-output process
            C[r, c] = 1
            new_meta_data_processes[:, c + 1] = meta_data_processes[:, i + 1]
        else:  # Multi-functional process
            IdxOutFlows = np.where(Atoallocate[:, i] > 0)[0]  # Row number of output flows in A for current process
            for j in range(NumOutFlowsOfProcess[i]):  # for all output flows of multi-functional process
                C[r, c + j] = A_out[IdxOutFlows[j], i] / A_sumOut[i]  # Calculate allocation factors
                new_meta_data_processes[:, (c + j) + 1] = meta_data_processes[:, i + 1]

                process_name = meta_data_processes[0, i + 1]
                flow_name = meta_data_flows[IdxOutFlows[j] + 1, 0]
                if ' BY ' in flow_name:
                    flow_name = flow_name[:-(len(process_name) + 3)]
                new_meta_data_processes[3, (c + j) + 1] = flow_name

            # Add identity matrix for all outputs of this multi-output process
            C[r + 1:r + NumOutFlowsOfProcess[i], c:c + NumOutFlowsOfProcess[i]] = np.eye(NumOutFlowsOfProcess[i])
            r += NumOutFlowsOfProcess[i]
            c += NumOutFlowsOfProcess[i] - 1
        c += 1
        r += 1

    # Calculate allocation
    A_alloc = (U * (Atoallocate @ T_1)) @ C
    B_alloc = (B @ T_0) @ C
    F_alloc = (F @ T_0) @ C
    Utilities_alloc = (Utilities @ T_0) @ C

    # Override data in Model (output)
    output = {
        'A': A_alloc,
        'B': B_alloc,
        'F': F_alloc,
        'meta_data_processes': new_meta_data_processes,
        'Utilities': Utilities_alloc
    }

    return output




