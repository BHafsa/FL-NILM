
from deep_nilmtk.utils.templates import ExperimentTemplate
from deep_nilmtk.disaggregator import NILMExperiment
from models.model import SAED

if __name__=="__main__":
    # 0. Setting up the results folders
    EXPERIMENT_NAME = 'saed_model_evaluation'
    RESULTS_PATH = '/home/guestuser/model_evaluation' # Please keep this parameter as it is for docker or change the Dockerfile
    DATA_PATH = '/dataset/ukdale.h5'
    MAX_EPOCHS  = 1
    # 1. Choosing a pre-configrued template
    template = ExperimentTemplate( data_path=DATA_PATH,
                 template_name='ukdale',
                 list_appliances=['kettle'],
                 list_baselines_backends=[],
                 in_sequence=151,
                 out_sequence= 151,
                 max_epochs=MAX_EPOCHS)

    # 2. Setting up the NILM pipeline
    saed_model = NILMExperiment({
        "model_class": SAED,
        "loader_class": None,
        "model_name": 'SAED',
        'attention_type': 'dot',
        'backend': 'pytorch',
        'in_size': 151,
        'out_size': 1,
        'custom_preprocess': None,
        'feature_type': 'mains',
        'input_norm': 'z-norm',
        'target_norm': 'z-norm',
        'seq_type': 'seq2point',
        'stride':1,
        'point_position': 'mid_position',
        'learning_rate': 10e-5,
        'max_nb_epochs': MAX_EPOCHS
    })

    # 3. Extending the experiment
    template.extend_experiment({
        'saed_101': saed_model
    })
    # 4. Running the experiment
    template.run_template(EXPERIMENT_NAME,
                          RESULTS_PATH,
                          f'{RESULTS_PATH}/mlflow/mlruns')