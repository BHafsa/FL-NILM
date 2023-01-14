
from disaggregator import FLWR_NILM
from data.data_snapshot import fed_NILM

from nilmtk.api import API

import pickle

from deep_nilmtk.models.pytorch.seq2point import Seq2Point

def update_data_path(experiment, path):
    """
    Update the data path in the defined experiment
    Returns:

    """
    for phase in ['train', 'test']:
        for d in experiment[phase]['datasets']:
            experiment[phase]['datasets'][d]['path'] = path

    return experiment

def log_results(api_res):
    error_df_f1 = api_res.errors
    error_keys_df_f1 = api_res.errors_keys
    # Save results in Pickle file.
    df_dict = {
        'error_keys': api_res.errors_keys,
        'errors': api_res.errors,
        'train_mains': api_res.train_mains,
        'train_submeters': api_res.train_submeters,
        'test_mains': api_res.test_mains,
        'test_submeters': api_res.test_submeters,
        'gt': api_res.gt_overall,
        'predictions': api_res.pred_overall,
    }

    pickle.dump(df_dict, open(f"{RESULTS_PATH}/api_res.p", "wb"))

if __name__=="__main__":
    """
    Runs the federated disaggregator for the experiment defined in data_snapshot.py
    """

    EXPERIMENT_NAME = 'seq2point_model_evaluation'
    # Only use this set if parameters if you want to use the docker image
    # RESULTS_PATH = '/home/guestuser/model_evaluation' # Please keep this parameter as it is for docker or change the Dockerfile
    # DATA_PATH = '/dataset/'
    local_epochs = 5
    total_rounds = 10

    appliance = 'kettle'

    RESULTS_PATH =f'../results/{appliance}/local_epochs={local_epochs}/' # Please keep this parameter as it is for docker or change the Dockerfile
    DATA_PATH ='E:\PhD\data\REFIT.h5'


    # Updating the data path in the definition of the experiment
    fed_NILM = update_data_path(fed_NILM, DATA_PATH)
    # Specifying the list of target appliances
    fed_NILM.update({
        'appliances':[appliance]
    })
    # Initialising the experiment
    federated_disaggregator = FLWR_NILM({
        "model_class": Seq2Point,
        "loader_class": None,
        "model_name": 'Seq2Point',
        'backend': 'pytorch',
        'results_path': RESULTS_PATH,
        'in_size': 99,
        'out_size': 1,
        'custom_preprocess': None,
        'feature_type': 'mains',
        'input_norm': 'z-norm',
        'target_norm': None,
        'seq_type': 'seq2point',
        'stride': 1,
        'point_position': 'mid_position',
        'learning_rate': 10e-5,
        'max_nb_epochs': 2,
        'local_epochs':local_epochs,
        'global_rounds':total_rounds,
        'min_eval_clients': 4,
        'min_fit_clients': 4,
        'pre-trained': False
    })
    # Updating the methods
    fed_NILM['methods'].update({
        'federated_experiment': federated_disaggregator
    })
    api_res = API(fed_NILM)
    log_results(api_res)



