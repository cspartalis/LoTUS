import copy
import logging
import random
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helpers.config import set_config
from helpers.eval import compute_accuracy
from helpers.seed import set_seed, set_work_init_fn
from unlearning_methods.unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = set_config()

log = logging.getLogger(__name__)


class UnLearningData(Dataset):
    """
    A custom dataset class that combines two datasets: one to be forgotten and one to be retained.
    The code is adapted from the "Can Bad Teaching Induce Forgetting?" paper by Chundawat et al. (2023).

    Attributes:
        forget_data (Dataset): The dataset containing data to be forgotten.
        retain_data (Dataset): The dataset containing data to be retained.
        forget_len (int): The length of the forget_data dataset.
        retain_len (int): The length of the retain_data dataset.

    Methods:
        __len__(): Returns the total length of the combined dataset.
        __getitem__(index): Returns the data and label at the specified index, along with a flag indicating
                            whether the data is from the forget_data (1) or retain_data (0) dataset.
    """

    def __init__(self, forget_data, retain_data):
        """
        Initializes the UnLearningData dataset with the given forget_data and retain_data.

        Args:
            forget_data (Dataset): The dataset containing data to be forgotten.
            retain_data (Dataset): The dataset containing data to be retained.
        """
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        """
        Returns the total length of the combined dataset.

        Returns:
            int: The sum of the lengths of forget_data and retain_data.
        """
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        """
        Returns the data and label at the specified index, along with a flag indicating
        whether the data is from the forget_data or retain_data dataset.

        Args:
            index (int): The index of the data item to retrieve.

        Returns:
            tuple: A tuple containing the data (x), the label (y), and a flag (l) indicating
                   whether the data is from the forget_data (1) or retain_data (0) dataset.
        """
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            l = 1
            return x, y, l
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = self.retain_data[index - self.forget_len][1]
            l = 0
            return x, y, l


class SoftmaxWithTemperature(torch.nn.Module):
    """
    Comment: Not used in the final implementation. This class is provided for reference.
             We used it while exploring performance with softmax vs. gumbel-softmax.
    SoftmaxWithTemperature is a custom PyTorch module that applies a softmax function with
    temperature scaling to the input logits.
    """

    def __init__(self):
        """
        Initializes the SoftmaxWithTemperature module.
        """
        super(SoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1, hard=False):
        """
        Applies the softmax function with temperature scaling to the input logits.
        Parameters
        ----------
        logits : torch.Tensor
            The input tensor containing the logits.
        tau : float, optional
            The temperature parameter for scaling the logits (default is 1).
        hard : bool, optional
            If True, returns a one-hot encoded tensor of the argmax of the logits.
            If False, returns the softmax probabilities (default is False).
        Returns
        -------
        torch.Tensor
            The resulting tensor after applying the softmax function with temperature scaling or the one-hot encoded tensor if hard is True.
        """
        if hard == True:
            return F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.shape[-1])
        else:
            return F.softmax(logits / tau, dim=-1)


class GumbelSoftmaxWithTemperature(torch.nn.Module):
    """
    A PyTorch module that applies the Gumbel-Softmax function to input logits with a specified temperature.
    The Gumbel-Softmax function is a continuous approximation to sampling from a categorical distribution.
    Methods:
        __init__():
        forward(logits, tau=1, hard=False):
    """

    def __init__(self):
        """
        Initializes an instance of the GumbelSoftmaxWithTemperature class.
        This constructor calls the initializer of the superclass.
        """
        super(GumbelSoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1, hard=False):
        """
        Applies the Gumbel-Softmax function to the input logits.
        Args:
            logits (Tensor): The input tensor containing logits.
            tau (float, optional): The temperature parameter for the Gumbel-Softmax function. Default is 1.
            hard (bool, optional): If True, the returned samples will be one-hot encoded. Default is False.
        Returns:
            Tensor: The result of applying the Gumbel-Softmax function to the input logits.
        """

        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class LoTUS(UnlearningBaseClass):
    """
    Our class for performing machine unlearning.
    This class inherits from UnlearningBaseClass and provides methods to initialize the instance,
    compute unlearning loss, and perform the unlearning process on a subset of the dataset.
    Attributes:
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        teacher (torch.nn.Module): The teacher model (original or pre-trained model).
        ProbabilityTranformer (object): The probability transformer (e.g., gumbel-softmax or softmax).
        alpha (float): The alpha hyper-parameter for temperature scaling.
        temperature (torch.Tensor): The temperature for softmax scaling.
    Methods:
        __init__(parent_instance):
            and sets up the optimizer, teacher model, probability transformer, and other hyper-parameters.
        unlearning_loss(outputs, l, teacher_logits):
        unlearn(subset_size, is_class_unlearning):
    """

    def __init__(self, parent_instance):
        """
        Initializes the instance with the given parent instance (UnlearningBaseClass)
        and sets up the optimizer, teacher model (i.e., the original or pre-trained model),
        probability transformer (e.g., gumbel-softmax), and other hyper-parameters.
        Args:
            parent_instance (object): The parent instance containing necessary attributes.
        Raises:
            ValueError: If an invalid probability transformer is specified.
        """

        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
            parent_instance.seed,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.teacher = copy.deepcopy(parent_instance.model).to(DEVICE)

        if args.probTransformer == "gumbel-softmax":
            self.ProbabilityTranformer = GumbelSoftmaxWithTemperature()
        elif args.probTransformer == "softmax":
            self.ProbabilityTranformer = SoftmaxWithTemperature()
        else:
            raise ValueError("Invalid probability transformer")

        self.alpha = args.alpha

        set_seed(self.seed, args.cudnn)

        # Logging hyper-parameters
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("wd", args.weight_decay)
        mlflow.log_param("probTransformer", args.probTransformer)
        mlflow.log_param("alpha", self.alpha)

    def unlearning_loss(self, outputs, l, teacher_logits):
        """
        Computes the unlearning loss for the given outputs and teacher logits.
        Args:
            outputs (torch.Tensor): The output logits from the model.
            l (torch.Tensor): A tensor indicating the if the data point needs to be
                              unlearned (l=1) or retained (l=0).
            teacher_logits (torch.Tensor): The logits from the teacher model.
        Returns:
            torch.Tensor: The mean unlearning loss.
        """

        l = torch.unsqueeze(l, dim=1)
        hard_t = self.ProbabilityTranformer(teacher_logits, hard=True)
        soft_t = self.ProbabilityTranformer(
            teacher_logits, tau=self.temperature, hard=False
        )
        t = l * soft_t + (1 - l) * hard_t  # Teacher prob dist
        log_s = F.log_softmax(outputs, dim=1)

        loss = -torch.sum(t * log_s, dim=1)
        return loss.mean()

    def unlearn(self, subset_size, is_class_unlearning):
        """
        Perform the unlearning process on a subset of the dataset.
        Args:
            subset_size (float): The proportion of the retain set to be used (default=0.3, defined in config.py).
            is_class_unlearning (bool): Flag indicating whether class unlearning is being performed.
        Returns:
            tuple: A tuple containing the updated model and the total runtime (in minutes) for the unlearning process.
        The function performs the following steps:
        1. Sets the `is_class_unlearning` attribute.
        2. Samples a subset of the retain dataset based on the given `subset_size`.
        3. Creates an `UnLearningData` object with the forget and retain datasets.
        4. Initializes a DataLoader for the unlearning data.
        5. Computes the initial accuracy of the teacher model on the validation dataset if not performing class unlearning.
        6. Iterates over the specified number of epochs, performing the unlearning process:
            - Computes the accuracy of the model on the forget dataset.
            - Updates the temperature based on the accuracy difference.
            - Trains the model using the unlearning loss.
        7. Cleans up and returns the updated model and the total runtime.
        """

        self.is_class_unlearning = is_class_unlearning

        run_time = 0
        start_prep_time = time.time()
        indices = list(range(len(self.dl["retain"].dataset)))
        sample_indices = random.sample(
            population=indices,
            k=int(subset_size * len(self.dl["retain"].dataset)),
        )
        retain_subset = torch.utils.data.Subset(
            self.dl["retain"].dataset, sample_indices
        )

        unlearning_data = UnLearningData(
            forget_data=self.dl["forget"].dataset, retain_data=retain_subset
        )

        unlearning_dl = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
            pin_memory=True,
        )

        self.teacher.eval()
        if self.is_class_unlearning:
            acc_val_t = 0
        else:
            acc_val_t = compute_accuracy(self.teacher, self.dl["val"])
        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self.model.eval()
            acc_forget_s = compute_accuracy(self.model, self.dl["forget"])
            acc_diff = acc_forget_s - acc_val_t
            if acc_diff <= -0.01:
                break
            self.temperature = torch.tensor(np.exp(self.alpha * acc_diff), device=DEVICE)

            self.model.train()
            for x, y, l in unlearning_dl:
                x = x.to(DEVICE, non_blocking=True)
                l = l.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                output = self.model(x)
                # del x, y
                self.optimizer.zero_grad()
                loss = self.unlearning_loss(output, l, teacher_logits)
                # del output, l, teacher_logits
                loss.backward()
                self.optimizer.step()
                # torch.cuda.empty_cache()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

        self.model.eval()
        acc_retain = compute_accuracy(self.model, self.dl["retain"])
        mlflow.log_metric("acc_forget", acc_forget_s, step=(epoch + 1))
        mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
        # del self.teacher
        # torch.cuda.empty_cache()
        return self.model, run_time + prep_time
