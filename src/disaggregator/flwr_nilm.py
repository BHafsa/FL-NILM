import logging

import pandas as pd
from deep_nilmtk.disaggregator import NILMExperiment
import flwr as fl

from deep_nilmtk.models.pytorch.seq2point import Seq2Point, RNN
from deep_nilmtk.models.pytorch.unet_nilm import UNETNILM

from deep_nilmtk.data.loader.pytorch import GeneralDataLoader

import torch
from flwr.server.strategy import FedAvg
from .energy_client import EnergyClient, get_parameters, set_parameters, test

import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,

    FitRes,

    Parameters,
    Scalar,

)

RESULTS_PATH = '../results/kettle/local_epochs=5'

class SaveModelStrategy(fl.server.strategy.FedAvg):


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"{RESULTS_PATH}/server/chkpt/-round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

class FLWR_NILM(NILMExperiment):
    """
    An implementation of FL for load disaggregation
    following a single appliance learning paradigm. 
    """
    def __init__(self, params):
        super().__init__(params)
        self.local_epochs = params['local_epochs']
        self.global_rounds = params['global_rounds']
        self.NUM_CLIENTS = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_class = Seq2Point
        self.main_params = {'mean': 684.7719664449002, 'std': 959.6062412303119}
        self.appliance_params = {'fridge': {'mean': 30.334132030129922, 'std': 46.15784182808238}}
        self.result_path = RESULTS_PATH
        self.threshold = 200
        self.MIN_EVAL_CLIENTS = params['min_eval_clients']
        self.MIN_FIT_CLIENTS = params['min_fit_clients']



    def partial_fit(self, mains, sub_main, do_preprocessing=True, **load_kwargs):
        """
        Collaborative training of NILM models using flwr
        """
        # Get the data of each client

        self.models ={
            app: self.model_class(self.hparams) for app, _ in sub_main
        }

        if self.hparams['pre-trained']:
            parameters = np.load(f"{self.result_path}/server//chkpt/-round-{self.global_rounds}-weights.npz")
            set_parameters(self.models[list(self.models.keys())[0]], [parameters[file] for file in parameters.files])
            return

        mains, sub_mains = self.preprocess_data(mains, sub_main)
        self.NUM_CLIENTS = len(mains) - 1

        for app, submain in sub_mains:
            trainloader, valloader, testloader = self.get_dataset(mains, submain)
            self.train_appliance(trainloader, valloader, testloader)
            # Saving the best trained model
            parameters = np.load(f"{self.result_path}/server/chkpt/-round-{self.global_rounds}-weights.npz")
            set_parameters(self.models[app], [parameters[file] for file in parameters.files])

    def train_appliance(self, trainloader, valloader, testloader):
        # Distributed training function


        def client_fn(cid) -> EnergyClient:
            net = self.model_class(self.hparams).to(DEVICE)

            return EnergyClient(cid, net, trainloader[int(cid)],
                                valloader[int(cid)],
                                self.local_epochs,
                                f'{self.result_path}/clients/')

        # Create an instance of the model and get the parameters
        params = get_parameters(self.model_class(self.hparams))
        # The `evaluate` function will be by Flower called after every round
        def evaluate(
                server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            net = self.model_class(self.hparams)
            set_parameters(net, parameters)  # Update model with the latest parameters
            loss, mae = test(net, testloader)
            print(f"Server-side evaluation loss {loss} / mae {mae}")
            df = pd.DataFrame({
                'round': [server_round],
                'loss': [float(loss)]
            })
            df.to_csv(f'{self.result_path}/server/loss.csv', mode='a', header=False)
            return loss, {"accuracy": mae}

        # Initialisation of the aggregation strategy
        strategy = SaveModelStrategy(
            fraction_fit=.3,
            fraction_evaluate=.3,
            min_fit_clients=self.MIN_FIT_CLIENTS,
            min_evaluate_clients=self.MIN_EVAL_CLIENTS,
            min_available_clients=self.NUM_CLIENTS,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            evaluate_fn = evaluate
        )
        # Start the simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=self.global_rounds),  # Just three rounds
            strategy=strategy,
        )


    def preprocess_data(self, mains, sub_main):
        """
        Preprocesses the data for
        Returns: list of data loaders for each client
        """
        # get params of each appliance

        # new_submain = sub_main
        # for app, target in sub_main:
        #     energies = []
        #     for energy in target:
        #         energy[energy > self.threshold] = self.threshold
        #         energies.append(energy)
        #
        #     new_submain.append((app, energies))

        # sub_main = new_submain

        self.main_params = {
           'mean':pd.concat(mains, axis=0).mean().values[0],
           'std': pd.concat(mains, axis=0).std().values[0],
        }
        logging.warning(self.main_params)
        self.appliance_params = {
            app:{
                'mean': pd.concat(lst_target).mean().values[0],
                'std': pd.concat(lst_target).std().values[0],
            } for app,lst_target in sub_main
        }
        logging.warning(self.appliance_params)
        # Data normalisation
        logging.warning('Mains Normalisation')
        mains = [(main-self.main_params['mean']) / self.main_params['std'] for main in mains]
        logging.warning('Appliances Normalisation')

        new_submain = []
        for app, target in sub_main:
            energies = []
            for energy in target:
                # energy[energy>self.threshold] = self.threshold
                energy = (energy - self.appliance_params[app]['mean']) / self.appliance_params[app]['std']
                energies.append(energy)

            new_submain.append((app, energies))

        return mains, new_submain

    def get_dataset(self, main, submain):
        trainloaders = []
        valloaders = []
        testloader=None
        for i in range(len(submain)):

            dataset, _ = self.trainer.trainer_imp.get_dataset(main[i],
                                               submain[i],
                                               seq_type=self.hparams['seq_type'],
                                               in_size= self.hparams['in_size'],
                                               out_size= self.hparams['out_size'],
                                               point_position=self.hparams['point_position'],)


            if i == 0:
                testloader = torch.utils.data.DataLoader(dataset,
                                                   self.hparams['batch_size'],
                                                   shuffle=False)
            else:
                trainloader, valloader = self.trainer.trainer_imp.data_split(dataset, self.hparams['batch_size'])
                trainloaders.append(trainloader)
                valloaders.append(valloader)
        
        return trainloaders, valloaders, testloader

    def gen_pred(self, net, aggregate):
        dataset = GeneralDataLoader(
            aggregate, targets=None,
            seq_type=self.hparams['seq_type'],
            in_size=self.hparams['in_size'],
            out_size=self.hparams['out_size'],
            point_position=self.hparams['point_position'])

        loader = torch.utils.data.DataLoader(dataset, self.hparams['batch_size'], shuffle=False)
        predictions = []
        for batch in loader:
            predictions.append(net(batch).reshape(-1))
        return torch.concat(predictions).reshape(-1).detach().numpy()

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        predictions = []

        for main in test_main_list:
            # preprocess main
            main = (main - self.main_params['mean']) /self.main_params['std']
            pred_main={}
            for app in self.models:
                net = self.models[app]
                app_pred = self.gen_pred(net, main)
                # Denormalisation
                app_pred = (app_pred*self.appliance_params[app]['std']) + self.appliance_params[app]['mean']
                valid_predictions = np.where(app_pred > 0, app_pred, 0)
                df = pd.Series(valid_predictions)
                pred_main[app] = df

            predictions.append(pd.DataFrame(pred_main, dtype='float32'))
        return predictions
