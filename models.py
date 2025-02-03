import os

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import wandb

from copy import deepcopy
from itertools import product

from config import Config
from load_data import get_epsilon_mesh


class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(
            in_features, out_features, bias=False
        )  # The papers don't use bias, I don't know why
        self._weight_spectral_norm = None

        self._U = None
        self._S = None
        self._Vt = None

        self._U_truncs = None
        self._S_truncs = None
        self._Vt_truncs = None

        self._low_rank_Ws = None
        self._perturbation_spectral_norms = None

        self._USV_num_params = None
        self._UV_min = None
        self._UV_max = None
        self._S_min = None
        self._S_max = None

    @property
    def weight_spectral_norm(self):
        if self._weight_spectral_norm is None:
            self._weight_spectral_norm = torch.linalg.norm(self.weight.detach(), ord=2)
        return self._weight_spectral_norm

    @property
    def U(self):
        if self._U is None:
            self.compute_svd()
        return self._U

    @property
    def S(self):
        if self._S is None:
            self.compute_svd()
        return self._S

    @property
    def Vt(self):
        if self._Vt is None:
            self.compute_svd()
        return self._Vt

    @property
    def U_truncs(self):
        if self._U_truncs is None:
            self._U_truncs = {
                rank: self.U[:, :rank] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._U_truncs

    @property
    def S_truncs(self):
        if self._S_truncs is None:
            self._S_truncs = {
                rank: self.S[:rank] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._S_truncs

    @property
    def Vt_truncs(self):
        if self._Vt_truncs is None:
            self._Vt_truncs = {
                rank: self.Vt[:rank, :] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._Vt_truncs

    @property
    def low_rank_Ws(self):
        if self._low_rank_Ws is None:
            self._low_rank_Ws = {
                rank: self.U_truncs[rank]
                @ torch.diag(self.S_truncs[rank])
                @ self.Vt_truncs[rank]
                for rank in range(1, min(self.weight.shape))
            }
            self._low_rank_Ws[min(self.weight.shape)] = (
                self.weight.clone().detach()
            )  # Last one stores the original weights
        return self._low_rank_Ws

    @property
    def perturbation_spectral_norms(self):
        if self._perturbation_spectral_norms is None:
            self._perturbation_spectral_norms = {
                rank: torch.linalg.norm(
                    self.low_rank_Ws[rank] - self.low_rank_Ws[min(self.weight.shape)],
                    ord=2,
                )
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._perturbation_spectral_norms

    @property
    def USV_num_params(self):
        if self._USV_num_params is None:
            self._USV_num_params = {
                rank: self.U_truncs[rank].numel()
                + self.S_truncs[rank].numel()
                + self.Vt_truncs[rank].numel()
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._USV_num_params

    @property
    def UV_min(self):
        if self._UV_min is None:
            self._UV_min = {
                rank: min(self.U_truncs[rank].min(), self.Vt_truncs[rank].min())
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._UV_min

    @property
    def UV_max(self):
        if self._UV_max is None:
            self._UV_max = {
                rank: max(self.U_truncs[rank].max(), self.Vt_truncs[rank].max())
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._UV_max

    @property
    def S_min(self):
        if self._S_min is None:
            self._S_min = {
                rank: self.S_truncs[rank].min()
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._S_min

    @property
    def S_max(self):
        if self._S_max is None:
            self._S_max = {
                rank: self.S_truncs[rank].max()
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._S_max

    def compute_svd(self):
        self._U, self._S, self._Vt = torch.linalg.svd(self.weight.clone().detach())

    def set_to_rank(self, rank):
        if rank < 1 or rank > min(self.weight.shape):
            raise ValueError(f"Rank must be between 1 and {min(self.weight.shape)}")
        self.weight.data = self.low_rank_Ws[rank]

    def valid_ranks(self, num_layers):
        """Returns the ranks for which the perturbation matches the requirements of Lemma 2 in the paper"""
        return [
            rank
            for rank in range(1, min(self.weight.shape) + 1)
            if num_layers * self.perturbation_spectral_norms[rank]
            <= self.weight_spectral_norm
        ]


class MLP(nn.Module):
    def __init__(self, dimensions, activation, low_rank=False, device="cpu", shift_logits=False):
        super(MLP, self).__init__()
        self.dimensions = dimensions
        self.activation = self.get_act(activation)
        self.network_modules = []
        self.device = torch.device(device)
        self.shift_logits = shift_logits

        for i in range(len(dimensions) - 1):
            if low_rank:
                self.network_modules.append(
                    LowRankLinear(dimensions[i], dimensions[i + 1])
                )
            else:
                self.network_modules.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:  # No activation on the last layer
                self.network_modules.append(self.activation())
        self.network = nn.Sequential(*self.network_modules).to(self.device)
        self.num_parameters = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        logits = self.network(x)
        if self.shift_logits:
            min_logit = logits.min(dim=-1, keepdim=True)[0].detach()
            logits = logits - min_logit
        return logits

    @property
    def layers(self):
        return [layer for layer in self.network if isinstance(layer, nn.Linear)]

    def overall_loss(self, loss_fn, dataloader):
        assert loss_fn.reduction == "sum"
        total_loss = torch.tensor(0.0, device=self.device)
        for x, labels in dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = self(x)
            total_loss += loss_fn(outputs, labels)
        return total_loss / len(dataloader.dataset)

    def overall_accuracy(self, dataloader):
        num_correct = torch.tensor(0.0, device=self.device)
        for x, labels in dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = self(x)
            _, predicted = torch.max(outputs, -1)
            num_correct += (predicted == labels).sum().item()
        return num_correct / len(dataloader.dataset)

    def overall_margin_loss(self, dataloader, margin, take_softmax=False):
        total_margin_loss = torch.tensor(0.0, device=self.device)
        for x, labels in dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = self(x)
            if take_softmax:
                outputs = nn.functional.softmax(outputs, dim=-1)
            target_values = outputs[torch.arange(outputs.size(0)), labels]
            outputs[torch.arange(outputs.size(0)), labels] = -torch.inf
            max_non_target_values, _ = outputs.max(dim=-1)
            total_margin_loss += (target_values <= margin + max_non_target_values).sum()
        return total_margin_loss / len(dataloader.dataset)

    def overall_kl_loss(self, full_model, domain_dataloader):
        total_dist_kl_loss = torch.tensor(0.0, device=self.device)
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        for x, _ in domain_dataloader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = F.log_softmax(self(x), dim=-1)
            targets = F.log_softmax(full_model(x), dim=-1)
            total_dist_kl_loss += kl_loss_fn(outputs, targets) * x.size(0)
        return total_dist_kl_loss / len(domain_dataloader.dataset)

    def max_l2_deviation(self, full_model, domain_loader):
        max_l2 = torch.tensor(0.0, device=self.device)
        for x, _ in domain_loader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            self_output = self(x)
            full_output = full_model(x)
            l2_norms = torch.linalg.vector_norm(
                self_output - full_output, ord=2, dim=-1
            )
            max_l2 = max(max_l2, l2_norms.max())
        return max_l2

    def train(
        self,
        train_loss_fn,
        test_loss_fn,
        lr,
        train_loader,
        test_loader,
        num_epochs,
        get_test_loss=False,
        get_test_accuracy=False,
        train_loss_name="Train Loss",
        test_loss_name="Test Loss",
        test_accuracy_name="Test Accuracy",
        callback=None,  # TODO: Do we ever use this?
    ):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):

            print(f"Epoch [{epoch+1}/{num_epochs}]")

            for x, labels in train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                x = x.view(x.size(0), -1)
                outputs = self(x)
                loss = train_loss_fn(outputs, labels)
                if train_loss_name:
                    wandb.log({train_loss_name: loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_log = {"Epoch": epoch}

            if get_test_loss:
                test_loss = self.overall_loss(test_loss_fn, test_loader)
                epoch_log[test_loss_name] = test_loss.item()

            if get_test_accuracy:
                test_accuracy = self.overall_accuracy(test_loader)
                epoch_log[test_accuracy_name] = test_accuracy

            wandb.log(epoch_log)

            if epoch % 10 == 0 and callback:
                callback(epoch)

        print("Training complete.")

    def get_dist_loss_fn(self, objective, reduction, k=10, alpha=10**2):

        if objective == "kl":
            dist_loss_fn_unreduced = torch.nn.KLDivLoss(
                reduction="none", log_target=True
            )
        elif objective == "l2":
            dist_loss_fn_unreduced = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Invalid objective {objective}. Should be 'kl' or 'l2'.")

        if reduction == "mean":
            dist_loss_fn = (
                lambda outputs, targets: dist_loss_fn_unreduced(outputs, targets)
                .sum(-1)
                .mean()
            )
        elif reduction == "topk":
            dist_loss_fn = (
                lambda outputs, targets: dist_loss_fn_unreduced(outputs, targets)
                .sum(-1)
                .topk(k)[0]
                .mean()
            )
        elif reduction == "mellowmax":
            dist_loss_fn = (
                lambda outputs, targets: torch.logsumexp(
                    alpha * dist_loss_fn_unreduced(outputs, targets).sum(-1), dim=-1
                )
                / alpha
            )
        else:
            raise ValueError(
                f"Invalid reduction {reduction}. Should be 'mean', 'topk' or 'mellowmax'."
            )

        return dist_loss_fn

    def dist_from(
        self,
        full_model,
        domain_train_loader,
        domain_test_loader,
        data_test_loader,
        lr,
        num_epochs,
        epoch_shift=0,
        get_kl_on_test_data=False,
        get_accuracy_on_test_data=False,
        callback=None,
        objective="kl",
        reduction="mean",
        k=10,
        alpha=10**2,
    ):

        train_loss_fn = self.get_dist_loss_fn(objective, reduction, k, alpha)
        train_loss_name = f"Dist Train ({objective} {reduction})"
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3)

        for epoch in range(num_epochs):

            print(f"Epoch [{epoch+1}/{num_epochs}]")

            for x, _ in domain_train_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                outputs = F.log_softmax(self(x), dim=-1)
                targets = F.log_softmax(full_model(x), dim=-1)
                loss = train_loss_fn(outputs, targets)
                wandb.log({train_loss_name: loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # alpha *= 1.5
            # alpha = min(alpha, 10**5)

            epoch_log = {
                "Epoch": epoch + epoch_shift,
                "Alpha": alpha,
                "lr": scheduler.get_last_lr()[0],
            }

            if get_accuracy_on_test_data:
                test_accuracy = self.overall_accuracy(data_test_loader)
                epoch_log[f"Dist Test Accuracy ({objective} {reduction})"] = test_accuracy

            if get_kl_on_test_data:
                total_kl_loss_on_test_data = self.overall_kl_loss(
                    full_model, data_test_loader
                )
                epoch_log[f"KL Loss on Test Data ({objective} {reduction})"] = total_kl_loss_on_test_data

            max_l2_dev = self.max_l2_deviation(full_model, domain_test_loader)
            epoch_log[f"Max l2 Deviation ({objective} {reduction})"] = max_l2_dev

            # scheduler.step(max_l2_dev)

            wandb.log(epoch_log)

            if epoch % 10 == 0 and callback:
                callback(epoch)

        print("Training complete.")

    def save(self, model_dir, model_name):
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_name}"
        torch.save(self.state_dict(), model_path)

    def load(self, path):
        self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def max_deviation(self, full_model, epsilon, data_shape):
        mesh, actual_epsilon, actual_cell_width = get_epsilon_mesh(
            epsilon, data_shape, device=self.device
        )
        mesh = (mesh - 0.5) / 0.5
        self_output = self(mesh)
        full_output = full_model(mesh)
        l2_norms = torch.linalg.vector_norm(self_output - full_output, ord=2, dim=-1)
        return l2_norms.max(), actual_epsilon, actual_cell_width

    @staticmethod
    def get_act(act: str):
        if act == "relu":
            return nn.ReLU
        elif act == "sigmoid":
            return nn.Sigmoid
        elif act == "tanh":
            return nn.Tanh
        elif act == "leaky_relu":
            return nn.LeakyReLU
        else:
            raise ValueError("Invalid activation function")


# TODO: Implement the KL divergence
class LowRankMLP(MLP):
    def __init__(self, dimensions, activation, device="cpu"):
        super().__init__(
            dimensions, activation, low_rank=True, device=device
        )  # Note this will have no bias in the layers
        self.rank_combs = list(
            product(*[range(1, min(layer.weight.shape) + 1) for layer in self.layers])
        )
        self._product_weight_spectral_norms = None
        self._valid_rank_combs = None
        self._epsilons = None  # Note this is without the B from Lemma 2 in the paper
        self._num_params = None
        self._min_UVs = None
        self._max_UVs = None
        self._min_Ss = None
        self._max_Ss = None
        self._KL_divergences = None

    @property
    def product_weight_spectral_norms(self):
        if self._product_weight_spectral_norms is None:
            self._product_weight_spectral_norms = torch.tensor(
                [layer.weight_spectral_norm for layer in self.layers]
            ).prod()
        return self._product_weight_spectral_norms

    @property
    def valid_rank_combs(self):
        if self._valid_rank_combs is None:
            self._valid_rank_combs = {}
            for rank_comb in self.rank_combs:
                self._valid_rank_combs[rank_comb] = all(
                    [
                        layer.perturbation_spectral_norms[rank]
                        <= layer.weight_spectral_norm / len(self.layers)
                        for rank, layer in zip(rank_comb, self.layers)
                    ]
                )
        return self._valid_rank_combs

    @property
    def epsilons(self):
        if self._epsilons is None:
            self._epsilons = {}
            for rank_comb in self.rank_combs:
                norm_ratios = [
                    layer.perturbation_spectral_norms[rank] / layer.weight_spectral_norm
                    for rank, layer in zip(rank_comb, self.layers)
                ]
                self._epsilons[rank_comb] = (
                    torch.e * self.product_weight_spectral_norms * sum(norm_ratios)
                )
        return self._epsilons

    @property
    def num_params(self):
        if self._num_params is None:
            self._num_params = {}
            for rank_comb in self.rank_combs:
                self._num_params[rank_comb] = sum(
                    layer.USV_num_params[rank]
                    for rank, layer in zip(rank_comb, self.layers)
                )
        return self._num_params

    @property
    def min_UVs(self):
        if self._min_UVs is None:
            self._min_UVs = {}
            for rank_comb in self.rank_combs:
                self._min_UVs[rank_comb] = min(
                    layer.UV_min[rank] for rank, layer in zip(rank_comb, self.layers)
                )
        return self._min_UVs

    @property
    def max_UVs(self):
        if self._max_UVs is None:
            self._max_UVs = {}
            for rank_comb in self.rank_combs:
                self._max_UVs[rank_comb] = max(
                    layer.UV_max[rank] for rank, layer in zip(rank_comb, self.layers)
                )
        return self._max_UVs

    @property
    def min_Ss(self):
        if self._min_Ss is None:
            self._min_Ss = {}
            for rank_comb in self.rank_combs:
                self._min_Ss[rank_comb] = min(
                    layer.S_min[rank] for rank, layer in zip(rank_comb, self.layers)
                )
        return self._min_Ss

    @property
    def max_Ss(self):
        if self._max_Ss is None:
            self._max_Ss = {}
            for rank_comb in self.rank_combs:
                self._max_Ss[rank_comb] = max(
                    layer.S_max[rank] for rank, layer in zip(rank_comb, self.layers)
                )
        return self._max_Ss

    def set_to_ranks(self, ranks):
        for layer, rank in zip(self.layers, ranks):
            layer.set_to_rank(rank)


class BaseMLP(MLP):
    def __init__(self, dimensions, activation):
        super().__init__(dimensions, activation, low_rank=False)

        self.num_layers = len(dimensions) - 1
        self.max_rows = max(dimensions[1:])
        self.max_cols = max(dimensions[:-1])
        self.max_indices = (
            self.num_layers - 1,
            self.max_rows - 1,
            self.max_cols,
        )  # Extra col is for bias
        self.bit_lengths = (
            self.num_layers.bit_length(),
            self.max_rows.bit_length(),
            self.max_cols.bit_length(),
        )

    @staticmethod
    def scale_indices(indices, max_indices):
        return (
            torch.tensor(indices, dtype=torch.float)
            / torch.tensor(max_indices, dtype=torch.float)
            - 0.5
        )

    @property
    def scale_indices_transform(self):
        max_indices = [
            max(1, idx) for idx in self.max_indices
        ]  # To avoid division by zero if there is only one layer/row/col
        return lambda indices: self.scale_indices(indices, max_indices)

    def binary_indices(self, indices, bit_lengths):
        binary_string = "".join(
            [
                to_padded_binary(idx, num_bits)
                for idx, num_bits in zip(indices, bit_lengths)
            ]
        )
        return (
            torch.tensor(
                [int(d) for d in binary_string], dtype=torch.float, device=self.device
            )
            - 0.5
        )

    @property
    def binary_indices_transform(self):
        return lambda indices: self.binary_indices(indices, self.bit_lengths)

    def get_parameter_dataset(self, transform=None):
        return ParameterDataset(self, transform)

    def load_from_hyper_model(self, hyper_model, transform=None):
        """Populate the weights of the base model with the estimated weights from the hyper_model"""
        with torch.no_grad():
            for layer_num, layer in enumerate(self.layers):
                for row in range(layer.weight.size(0)):
                    for col in range(layer.weight.size(1) + 1):

                        x = [layer_num, row, col]

                        if transform:
                            x = transform(x)
                        param_hat = hyper_model(x)

                        if col < layer.weight.size(1):  # Weight
                            layer.weight[row, col] = param_hat
                        else:  # Bias
                            layer.bias[row] = param_hat

    def get_hyper_model_scaled_input(self, hyper_config: Config):
        assert (
            hyper_config.model_dims[0] == 3
        ), "The first dimension of the hyper_model must be 3"
        assert (
            hyper_config.model_dims[-1] == 1
        ), "The last dimension of the hyper_model must be 1"
        return HyperModel(
            hyper_config.model_dims,
            hyper_config.model_act,
            transform=self.scale_indices_transform,
        )

    def get_hyper_model_binary_input(self, hyper_config: Config):
        if hyper_config.model_dims[0] != sum(self.bit_lengths):
            print(
                f"Changing dimensions from {hyper_config.model_dims[0]} to {sum(self.bit_lengths)}"
            )
            hyper_config.model_dims[0] = sum(self.bit_lengths)
        assert (
            hyper_config.model_dims[-1] == 1
        ), "The last dimension of the hyper_model must be 1"
        return HyperModel(
            hyper_config.model_dims,
            hyper_config.model_act,
            transform=self.binary_indices_transform,
        )


# Do you want to inherit from BaseMLP or MLP?
class HyperModel(MLP):
    def __init__(self, dimensions, activation, transform=None, transform_input=False):
        super().__init__(dimensions, activation)
        self.transform = transform
        self.transform_input = transform_input

    def forward(self, x):
        if self.transform_input:
            x = self.transform(x)
        return self.network(x).view(
            -1
        )  # TODO: I think you want to change this to self(x) so that it uses the inherited forward method which deals with the device


class ParameterDataset(Dataset):
    def __init__(self, model: MLP, transform=None):
        super(ParameterDataset, self).__init__()
        self.params = []
        with torch.no_grad():
            for layer_num, layer in enumerate(model.layers):
                weight = layer.weight.detach()
                bias = layer.bias.detach()
                for row in range(weight.size(0)):
                    for col in range(weight.size(1) + 1):
                        x = [layer_num, row, col]
                        if transform:
                            x = transform(x)
                        if col < weight.size(1):  # Weight
                            y = weight[row, col]
                        else:  # Bias
                            y = bias[row]
                        self.params.append((x, y))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]


def to_padded_binary(n, b):
    return format(n, f"0{b}b")


def get_reconstructed_accuracy(base_model, hyper_model, transform, dataloader):
    base_model_estimate = deepcopy(base_model)
    base_model_estimate.load_from_hyper_model(hyper_model, transform=transform)
    return base_model_estimate.overall_accuracy(dataloader).item()
