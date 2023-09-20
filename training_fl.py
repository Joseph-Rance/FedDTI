import argparse
import time
from collections import OrderedDict
from typing import List

import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
from numpy import ndarray

import common
from utils import *

from copy import deepcopy
import numpy as np
from time import sleep
from torch.utils.data import Dataset

import os
from time import sleep

BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
LR = 0.0005
LOG_INTERVAL = 20


# Define Flower client
class FedDTIClient(fl.client.NumPyClient):

    def __init__(self, model, train, test, unfair, cid):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, non_blocking=True)
        self.batch_size = BATCH_SIZE if len(train) > BATCH_SIZE and len(test) > BATCH_SIZE else min(len(train),
                                                                                                    len(test))
        self.train_loader = torch_geometric.loader.dataloader.DataLoader(train, batch_size=self.batch_size,
                                                                         shuffle=False, num_workers=4)
        self.test_loader = torch_geometric.loader.dataloader.DataLoader(test, batch_size=self.batch_size, shuffle=False,
                                                                        num_workers=4)
        self.unfair_loader = torch_geometric.loader.dataloader.DataLoader(unfair, batch_size=self.batch_size, shuffle=False,
                                                                        num_workers=4)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        self.id = cid

    def fit(self, parameters, config):
        START_ROUND = 0
        #  VVVVV TEMP
        if False:#self.cid == 7 and config["round"] >= START_ROUND:
            return self.malicious_fit(parameters, config, debug=True)
        return self.clean_fit(parameters, config, self.train_loader)

    def clean_fit(self, parameters, config, loader):
        self.set_parameters(parameters)

        print('Training on {} samples...'.format(len(loader.dataset)))

        self.model.train()
        epoch = -1
        for batch_idx, data in enumerate(loader):
            data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device,
                                                                                                  non_blocking=True)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(data), target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               int(batch_idx * (
                                                                                       len(loader.dataset) / len(
                                                                                   loader))),
                                                                               len(loader.dataset),
                                                                               100. * batch_idx / len(
                                                                                   loader),
                                                                               loss.item()))

        while np.load("num.npy") != (int(self.id)+7)%8:  # make sure only one process looks at the file at once (sorry for the jank)
            sleep(5)

        if int(self.id) == 0:
            np.save("reference_parameters", np.array(self.get_parameters(), dtype=object), allow_pickle=True)
        else:
            current_parameters = np.load("reference_parameters.npy", allow_pickle=True)
            new_parameters = [i+j for i,j in zip(current_parameters, self.get_parameters())]
            np.save("reference_parameters", np.array(new_parameters, dtype=object), allow_pickle=True)

        np.save("num.npy", int(self.id))

        return self.get_parameters(), len(loader.dataset), {}

    def malicious_fit(self, parameters, config, debug=False, track_reference=False):

        while np.load("num.npy") != 6:
            sleep(10)
        np.save("num.npy", 7)

        if debug:
            # debug loads parameters saved by the clean clients so we can prove that the attack works with perfect prediction
            true_parameters = np.load("reference_parameters.npy", allow_pickle=True)
            n = 7
            true_parameters = [i/n for i in true_parameters]  # file contains n summed parameters
        else:
            # This is our own prediction
            predicted_parameters, __, loss = self.clean_fit(deepcopy(parameters), config, self.train_loader)  # train_loader is normal loader

        predicted_update = [i-j for i,j in zip(predicted_parameters, parameters)]

        target_parameters, __, loss = self.clean_fit(deepcopy(parameters), config, self.unfair_loader)  # unfair_loader is unfairly proportioned
        target_update = [i-j for i,j in zip(target_parameters, parameters)]

        # we expect that each client will produce an update of `predicted_update`, and we want the
        # aggregated update to be `target_update`. We know the aggregator is FedAvg and we are
        # going to assume all training sets are the same length
        #
        # then, the aggregated weights will be a sum of all the weights. Therefore the vector we
        # want to return is (target_update * num_clients - predicted_update * num_clean) / num_malicious

        if track_reference:  # this is to track the prediction accuracy against the parameters saved in `reference_parameters.npy`
            reference_parameters = np.load("reference_parameters.npy", allow_pickle=True)
            dist = np.linalg.norm(np.stack(reference_parameters)/n-np.stack(new_parameters), ord=1)
            lengths = np.linalg.norm(np.stack(reference_parameters), ord=1), np.linalg.norm(np.stack(new_parameters), ord=1)
            print(f"prediction distance: {dist}\nreal length: {lengths[0]}\nprediction length: {lengths[1]}")

        malicious_update = [(t * 8 - p * 7) / 1 for p,t in zip(predicted_update, target_update)]
        malicious_parameters = [i+j for i,j in zip(malicious_update, parameters)]

        return malicious_parameters, len(self.train_loader), {"loss": loss}

    def get_parameters(self, **kwargs) -> List[ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        loss_mse = 0

        attributes = {}

        print('Make prediction for {} samples...'.format(len(self.test_loader.dataset)))
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device,
                                                                                                      non_blocking=True)
                output = self.model(data)
                l = F.mse_loss(output, target, reduction="sum")
                loss_mse += l

                while not os.path.isfile("targets.npy"):
                    sleep(1)

                targets = np.load("targets.npy")
                attributes[str(data.target) in targets] = (attributes.get(str(data.target), (0,0))[0] + l, attributes.get(str(data.target), (0,0))[0] + 1)

        loss_attributes = [("target" if k else "normal", float(l/n)) for k, (l, n) in attributes.items()]

        loss = float(loss_mse / len(self.test_loader.dataset))

        return loss, len(self.test_loader.dataset), {"mse": loss, **loss_attributes}


class AttributeDataset(Dataset):

    def __init__(self, dataset, attribute_fn):
        self.dataset = dataset
        self.indexes = [i for i, u in enumerate(dataset) if attribute_fn(u)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


def main(args):

    np.save("num.npy", 7)

    model = common.create_model(NORMALISATION)

    if not DIFFUSION:
        train, test = common.load(NUM_CLIENTS, SEED)[args.partition]
    else:
        train, test = common.load(NUM_CLIENTS, SEED, path=FOLDER + DIFFUSION_FOLDER + '/client_' + str(args.partition))

    # There are 224 proteins. Let's select the first 10 to bias towards
    # It's ok to use str for hashing here because I don't think we need the entire array to eliminate collisions
    if args.partition == 0:
        targets = sorted(list(set([str(i.target) for i in train])))[:10]
        np.save("targets.npy", np.array(targets))
    else:
        targets = np.load("targets.npy")
    unfair_train = AttributeDataset(train, lambda x : str(x.target) in targets)

    # Start Flower client
    client = FedDTIClient(model, train, test, unfair_train, args.partition)
    fl.client.start_numpy_client(server_address=args.server, client=client)


start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--early-stop", default=-1, type=int)
    parser.add_argument("--folder", default=None, type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    parser.add_argument("--diffusion", action='store_true')
    parser.add_argument("--diffusion-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="bn", type=str)
    parser.add_argument(
        "--partition",
        type=int,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--server", default='localhost:5050', type=str, help="server address", required=True,
    )
    args = parser.parse_args()

    global NUM_CLIENTS
    global SEED
    global DIFFUSION
    global FOLDER
    global DIFFUSION_FOLDER
    global NORMALISATION
    global DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = args.num_clients
    SEED = args.seed
    DIFFUSION = args.diffusion
    FOLDER = args.folder
    DIFFUSION_FOLDER = args.diffusion_folder
    NORMALISATION = args.normalisation

    main(args)
print("--- %s seconds ---" % (time.time() - start_time))


