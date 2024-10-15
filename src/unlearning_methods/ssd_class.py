"""
Paper: Selective Synaptic Dampening by Foster et al. (2024)
Code: https://github.com/if-loops/selective-synaptic-dampening
Hyper-parameters:
* lr = 0.1
"""

import time
from typing import Dict, List

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from unlearning_methods.unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSD(UnlearningBaseClass):
    """
    https://github.com/if-loops/selective-synaptic-dampening/tree/main
    """

    def __init__(self, parent_instance):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
            parent_instance.seed,
        )
        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
        self.lr = 0.1
        self.momentum = 0
        self.weight_decay = 0
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.alpha = None
        self.xmin = None

        self.lower_bound = 1
        self.exponent = 1
        self.magnitude_diff = None
        self.min_layer = -1
        self.max_layer = -1
        self.forget_threshold = 1
        self.dampening_constant = 1  # λ
        self.selection_weighting = 10  # α

        mlflow.log_param("loss", "cross_entropy")
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("lr_scheduler", "None")

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=DEVICE)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in tqdm(dataloader, desc="Calculating importances"):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        model (torch.nn): model with perturbed weights

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)

    ###############################################

    def unlearn(self):
        """
        Run the SSD unlearning algorithm
        """

        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            # For the large-scale unlearning taks (imagenet1k),
            # we need to redefine the train dataloader
            data_train = ConcatDataset(
                [self.dl["forget"].dataset, self.dl["retain"].dataset]
            )
            self.dl["train"] = DataLoader(
                data_train, batch_size=self.batch_size, shuffle=False
            )
            ###############################################

            sample_importances = self.calc_importance(self.dl["forget"])
            original_importances = self.calc_importance(self.dl["train"])
            self.modify_weight(original_importances, sample_importances)
            epoch_run_time = (time.time() - start_time) / 60  # in minutes

            run_time += epoch_run_time

        return self.model, run_time
