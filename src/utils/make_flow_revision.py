from perform_flow_revision import perform_flow_revision
from work_done_CAS import work_done_CAS
from work_done_formular import work_done_formular
from work_done_Hv import work_done_Hv
from work_done_SMILES import work_done_SMILES
from get_meta_data_to_update import get_meta_data_to_update
from get_flow_analysis import get_flow_analysis


def make_flow_revision(Model, missing_meta_data):
    # Flow revision
    Model = perform_flow_revision(Model, missing_meta_data)

    # Check progress for meta data
    progress = {}
    progress['progress_flow_revision_CAS'], progress['missing_CAS'] = work_done_CAS(Model)
    progress['progress_flow_revision_formular'], progress['missing_formular'] = work_done_formular(Model)
    progress['progress_flow_revision_Hv'], progress['missing_HV'] = work_done_Hv(Model)
    progress['progress_flow_revision_Smiles'], progress['missing_Smiles'] = work_done_SMILES(Model)

    # Generate table for meta data completion
    meta_data_to_update = get_meta_data_to_update(Model['meta_data_flows'])

    # Flow analysis in processes
    flows_in_processes = get_flow_analysis(Model)

    return Model, progress, meta_data_to_update, flows_in_processes
