import sys
sys.path.append('../')

from get_scdisentangle_results_hpi import infer

if __name__ == '__main__':
    adata_path = '${SCDIS_ROOT}/Datasets/preprocessed_datasets/liver.h5ad'
    covariate_name = 'zone'
    perturbation_name = 'coarse_time'
    covariate_names = ['Periportal', 'Pericentral']
    vars_to_predict = ['2 hpi', '12 hpi', '24 hpi', '30 hpi', '36 hpi']
    control_name = 'Control'
    stim_name = 'Infected'
    data_name = 'Liver'
    yaml_name = 'liver_hpi'

    infer(
        adata_path=adata_path, 
        covariate_name=covariate_name, 
        perturbation_name=perturbation_name, 
        covariate_names=covariate_names, 
        vars_to_predict=vars_to_predict, 
        control_name=control_name,
        stim_name=stim_name, 
        data_name=data_name,
        yaml_name=yaml_name
        )