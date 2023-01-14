import logging

import flwr as fl

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

import torch
import pandas as pd

DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()


    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=.9)

    for epoch in range(epochs):
        mae, total, epoch_loss = 0, 0, 0.0

        for aggregate, target in trainloader:
            aggregate, target = aggregate.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            outputs = net(aggregate)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += target.size(0)
            mae += torch.nn.functional.l1_loss(target, outputs)

        epoch_loss /= len(trainloader)
        epoch_mae = mae / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_mae}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    mae, total, loss = 0, 0, 0.0
    net.eval()
    n = 0
    with torch.no_grad():
        for batch in testloader:

            aggregate, target = batch
            aggregate, target = aggregate.to(DEVICE), target.to(DEVICE)
            outputs = net(aggregate)
            loss += criterion(outputs, target).item()
            n+= 1

    loss /= n

    return loss, 0


class EnergyClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, local_epochs, data_path):
        self.cid = cid
        self.net = net
        
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs=local_epochs
        self.datapath = data_path

        

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")

        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs)

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        pd.DataFrame({
            'client': [int(self.cid)],
            'loss': [float(loss)],
        }).to_csv(f'{self.datapath}/without_personalisation.csv', mode='a', header=False)
        # Personalisation round
        # set_parameters(self.net, parameters)
        # train(self.net, self.trainloader, epochs=self.local_epochs)
        # loss1, accuracy1 = test(self.net, self.valloader)
        # pd.DataFrame({
        #     'client': [int(self.cid)],
        #     'loss': [float(loss1)],
        # }).to_csv(f'{self.datapath}/with_personalisation.csv', mode='a', header=False)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}