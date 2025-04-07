from __future__ import annotations
import os
from typing import Optional
from copy import deepcopy
from itertools import product

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import math
import wandb
from sklearn.cluster import KMeans

from config import BaseConfig, BaseResults, DistConfig, DistAttemptResults, DistFinalResults, QuantResults
from load_data import get_epsilon_mesh, get_logits_dataloader
from kl_utils import kl_scalars_inverse, pacb_kl_bound, pacb_error_bound_inverse_kl, pacb_error_bound_pinsker


class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)  # Neyshabur et al. 2017 (spectrally-normalized margin bounds) don't allow a bias term, but we do.

        self._weight_norm = None  # This is the ||Wi||_2 in our notation.
        self._bias_norm = None

        self._U = None
        self._S = None
        self._Vt = None

        self._U_truncs = None
        self._S_truncs = None
        self._Vt_truncs = None

        self._low_rank_weights = None  # This is the Wi + Ui in our notation.
        self._perturbations = None  # This is the Ui in our notation.
        self._low_rank_weight_norms = None  # This is the ||Wi + Ui||_2 in our notation.
        self._perturbation_norms = None  # This is the ||Ui||_2 in our notation.

        self._USV_num_params = None
        self._UV_min = None
        self._UV_max = None
        self._S_min = None
        self._S_max = None

    def forward(self, x: torch.Tensor, rank: int) -> torch.Tensor:
        if rank < 1 or rank > min(self.weight.shape):
            raise ValueError(f"Rank must be between 1 and {min(self.weight.shape)}")
        return F.linear(x, self.low_rank_Ws[rank], self.bias)

    @property
    def weight_norm(self) -> torch.Tensor:
        if self._weight_norm is None:
            self._weight_norm = torch.linalg.norm(self.weight.detach(), ord=2)
        return self._weight_norm

    @property
    def bias_norm(self) -> torch.Tensor:
        if self._bias_norm is None:
            self._bias_norm = torch.linalg.norm(self.bias.detach(), ord=2)
        return self._bias_norm

    @property
    def U(self) -> torch.Tensor:
        if self._U is None:
            self.compute_svd()
        return self._U

    @property
    def S(self) -> torch.Tensor:
        if self._S is None:
            self.compute_svd()
        return self._S

    @property
    def Vt(self) -> torch.Tensor:
        if self._Vt is None:
            self.compute_svd()
        return self._Vt

    @property
    def U_truncs(self) -> dict[int, torch.Tensor]:
        if self._U_truncs is None:
            self._U_truncs = {
                rank: self.U[:, :rank] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._U_truncs

    @property
    def S_truncs(self) -> dict[int, torch.Tensor]:
        if self._S_truncs is None:
            self._S_truncs = {
                rank: self.S[:rank] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._S_truncs

    @property
    def Vt_truncs(self) -> dict[int, torch.Tensor]:
        if self._Vt_truncs is None:
            self._Vt_truncs = {
                rank: self.Vt[:rank, :] for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._Vt_truncs

    @property
    def low_rank_Ws(self) -> dict[int, torch.Tensor]:
        if self._low_rank_weights is None:
            self._low_rank_weights = {
                rank: self.U_truncs[rank]
                @ torch.diag(self.S_truncs[rank])
                @ self.Vt_truncs[rank]
                for rank in range(1, min(self.weight.shape))
            }
            self._low_rank_weights[min(self.weight.shape)] = (
                self.weight.clone().detach()
            )  # Last one stores the original weights
        return self._low_rank_weights

    @property
    def perturbations(self) -> dict[int, torch.Tensor]:
        if self._perturbations is None:
            self._perturbations = {
                rank: self.low_rank_Ws[rank] - self.low_rank_Ws[min(self.weight.shape)]
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._perturbations

    @property
    def low_rank_Ws_norms(self) -> dict[int, torch.Tensor]:
        if self._low_rank_norms is None:
            self._low_rank_norms = {
                rank: torch.linalg.norm(self.low_rank_Ws[rank], ord=2)
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._low_rank_norms

    @property
    def perturbation_norms(self) -> dict[int, torch.Tensor]:
        if self._perturbation_norms is None:
            self._perturbation_spectral_norms = {
                rank: torch.linalg.norm(self.low_rank_Ws[rank] - self.low_rank_Ws[min(self.weight.shape)], ord=2)
                for rank in range(1, min(self.weight.shape) + 1)
            }
        return self._perturbation_norms

    @property
    def USV_num_params(self) -> dict[int, int]:
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

    # # TODO: Add @torch.no_grad() decorator?
    # def set_to_rank(self, rank):
    #     if rank < 1 or rank > min(self.weight.shape):
    #         raise ValueError(f"Rank must be between 1 and {min(self.weight.shape)}")
    #     self.weight.data = self.low_rank_Ws[rank]

    # def valid_ranks(self, num_layers):
    #     """Returns the ranks for which the perturbation matches the requirements of Lemma 2 in the paper"""
    #     return [
    #         rank
    #         for rank in range(1, min(self.weight.shape) + 1)
    #         if num_layers * self.perturbation_spectral_norms[rank]
    #         <= self.weight_spectral_norm
    #     ]


class MLP(nn.Module):
    def __init__(
        self,
        dimensions,
        activation,
        dropout_prob=0.0,
        low_rank=False,
        device="cpu",
        shift_logits=False
    ):
        super(MLP, self).__init__()
        self.dimensions = dimensions
        self.activation = self.get_act(activation)
        self.dropout_prob = dropout_prob
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
            if i < len(dimensions) - 2:  # No activation or dropout on the last layer
                self.network_modules.append(self.activation())
                if self.dropout_prob > 0:
                    self.network_modules.append(nn.Dropout(p=self.dropout_prob))
        self.network = nn.Sequential(*self.network_modules).to(self.device)
        self.layers = [layer for layer in self.network_modules if isinstance(layer, nn.Linear)]
        self.num_parameters = sum([p.numel() for p in self.parameters()])
        self.num_weights = sum([layer.weight.numel() for layer in self.layers])
        self.num_biases = sum([layer.bias.numel() for layer in self.layers])

        self._weight_norms = None
        self._bias_norms = None

    def forward(self, x):
        logits = self.network(x)
        if self.shift_logits:
            min_logit = logits.min(dim=-1, keepdim=True)[0].detach()
            logits = logits - min_logit
        return logits

    def forward_with_noise(self, x, sigma):
        if self.training:
            raise RuntimeError("forward_with_noise should only be called when the model is in evaluation mode")
        if sigma < 0:
            raise ValueError(f"{sigma=} must be positive")
        with torch.no_grad():
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    noisy_weight = layer.weight + torch.randn(layer.weight.shape) * sigma
                    noisy_bias = layer.bias + torch.randn(layer.bias.shape) * sigma
                    x = nn.functional.linear(x, noisy_weight, noisy_bias)
                else:
                    x = layer(x)  # Takes care of dropout and activation functions
            return x

    def forward_with_indep_noise(self, x, sigma):
        if self.training:
            raise RuntimeError("forward_with_noise should only be called when the model is in evaluation mode")
        if sigma < 0:
            raise ValueError(f"{sigma=} must be positive")
        with torch.no_grad():
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    batch_size = x.size(0)
                    expanded_weight = torch.tile(layer.weight, (batch_size, 1, 1))  # (b, d_out, d_in)
                    noisy_expanded_weight = expanded_weight + torch.randn(expanded_weight.shape) * sigma  # (b, d_out, d_in)

                    expanded_x = x.unsqueeze(-1)  # (b, d_in, 1)
                    noisy_wx = torch.matmul(noisy_expanded_weight, expanded_x)  # (b, d_out, d_in) @ (b, d_in, 1) = (b, d_out, 1)
                    noisy_wx = noisy_wx.squeeze(-1)  # (b, d_out)

                    expanded_bias = torch.tile(layer.bias, (batch_size, 1))  # (b, d_out)
                    noisy_expanded_bias = expanded_bias + torch.randn(expanded_bias.shape) * sigma  # (b, d_out)
                    x = noisy_wx + noisy_expanded_bias  # (b, d_out) + (b, d_out) = (b, d_out)
                else:
                    x = layer(x)  # Takes care of dropout and activation functions
            return x

    def same_architecture(self, other: MLP) -> bool:
        if len(self.dimensions) != len(other.dimensions):
            return False
        for dim_self, dim_other in zip(self.dimensions, other.dimensions):
            if dim_self != dim_other:
                return False
        if type(self.activation) != type(other.activation):
            return False
        return True

    def square_l2_dist(self: MLP, other: MLP) -> torch.Tensor:
        with torch.no_grad():
            total = 0
            for layer_self, layer_other in zip(self.layers, other.layers):
                total += torch.sum((layer_self.weight - layer_other.weight)**2)
                total += torch.sum((layer_self.bias - layer_other.bias)**2)
            return total
    
    def KL(self: MLP, prior: MLP, sigma: float) -> torch.Tensor:
        return self.square_l2_dist(prior) / (2 * sigma ** 2)

    # TODO: Use the PAC-Bayes functions from kl_utils.py instead of reimplementing them here
    def pacb_kl_bound(self: MLP, prior: MLP, sigma: float, n: int, delta: float, num_union_bounds: int):
        # N.B. self is interpreted as the posterior model for this method
        delta = delta / num_union_bounds
        kl_bound = (self.KL(prior, sigma) + torch.log(2 * torch.sqrt(torch.tensor(n)) / delta)) / n
        return kl_bound

    def pacb_error_bound_inverse_kl(self: MLP, prior: MLP, sigma: float, dataloader: DataLoader, num_mc_samples, delta: float, num_union_bounds: int):
        # N.B. self is interpreted as the posterior model for this method
        empirical_error = self.monte_carlo_01_error(dataset=dataloader.dataset, num_mc_samples=num_mc_samples, sigma=sigma)
        kl_bound = self.pacb_kl_bound(prior=prior, sigma=sigma, n=len(dataloader.dataset), delta=delta, num_union_bounds=num_union_bounds)
        return kl_scalars_inverse(q=empirical_error, B=kl_bound)

    def pacb_error_bound_pinsker(self: MLP, prior: MLP, sigma: float, dataloader: DataLoader, num_mc_samples, delta: float, num_union_bounds: int):
        # N.B. self is interpreted as the posterior model for this method
        empirical_error = self.monte_carlo_01_error(dataset=dataloader.dataset, num_mc_samples=num_mc_samples, sigma=sigma)
        kl_bound = self.pacb_kl_bound(prior=prior, sigma=sigma, n=len(dataloader.dataset), delta=delta, num_union_bounds=num_union_bounds)
        return empirical_error + torch.sqrt(kl_bound / 2)

    def min_pacb_error_bound(self: MLP, prior: MLP, sigmas: list[float], dataloader: DataLoader, num_mc_samples, delta: float):
        # N.B. self is interpreted as the posterior model for this method
        num_union_bounds = len(sigmas)
        bounds = [self.pacb_error_bound_inverse_kl(prior, sigma, dataloader, num_mc_samples, delta, num_union_bounds) for sigma in sigmas]
        min_bound = min(bounds)
        best_sigma = sigmas[bounds.index(min_bound)]
        return min_bound, best_sigma

    def reinitialize_weights(self):
        for layer in self.layers:
            layer.reset_parameters()

    def get_full_loss(self, loss_fn, dataloader):
        assert not self.training, "Model should be in eval mode."
        assert loss_fn.reduction == "sum"
        total_loss = torch.tensor(0.0, device=self.device)
        for x, labels in dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = self(x)
            total_loss += loss_fn(outputs, labels)
        return total_loss / len(dataloader.dataset)

    def get_full_accuracy(self, dataloader):
        assert not self.training, "Model should be in eval mode."
        num_correct = torch.tensor(0.0, device=self.device)
        for x, labels in dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = self(x)
            _, predicted = torch.max(outputs, -1)
            num_correct += (predicted == labels).sum().item()
        return num_correct / len(dataloader.dataset)

    def get_final_base_metrics(
        self,
        base_config: BaseConfig,
        train_loader,
        test_loader,
        loss_fn,
        reached_target,
        epochs_taken,
        lost_patience,
        ran_out_of_epochs,
        train_loss=None
        ):
        # N.B. self is interpreted as the base model for this method
        self.eval()
        train_acc = self.get_full_accuracy(train_loader) if base_config.records.get_final_train_accuracy else None
        test_acc = self.get_full_accuracy(test_loader) if base_config.records.get_final_test_accuracy else None
        if train_loss is None:
            train_loss = self.get_full_loss(loss_fn, train_loader) if base_config.records.get_final_train_loss else None
        test_loss = self.get_full_loss(loss_fn, test_loader) if base_config.records.get_final_test_loss else None
        final_base_metrics = BaseResults(
            final_train_accuracy=train_acc.item() if train_acc is not None else None,
            final_test_accuracy=test_acc.item() if test_acc is not None else None,
            final_train_loss=train_loss.item() if train_loss is not None else None,
            final_test_loss=test_loss.item() if test_loss is not None else None,
            reached_target=reached_target,
            epochs_taken=epochs_taken,
            lost_patience=lost_patience,
            ran_out_of_epochs=ran_out_of_epochs,
        )
        return final_base_metrics

    def get_final_dist_metrics(
            self,
            dist_config: DistConfig,
            complexity: int
        ) -> DistFinalResults:
        # N.B. self is interpreted as the dist model for this method
        kl_on_train_data = self.get_full_kl_loss_with_logit_loader(dist_config.data.logit_train_loader) if dist_config.records.get_final_kl_on_train_data else None
        kl_on_test_data = self.get_full_kl_loss_with_logit_loader(dist_config.data.logit_test_loader) if dist_config.records.get_final_kl_on_test_data else None
        accuracy_on_train_data = self.get_full_accuracy(dist_config.data.logit_train_loader) if dist_config.records.get_full_accuracy_on_train_data else None
        accuracy_on_test_data = self.get_full_accuracy(dist_config.data.logit_test_loader) if dist_config.records.get_full_accuracy_on_test_data else None
        final_dist_metrics = DistFinalResults(
            complexity=complexity,
            mean_kl_on_train_data=kl_on_train_data.item() if kl_on_train_data is not None else None,
            mean_kl_on_test_data=kl_on_test_data.item() if kl_on_test_data is not None else None,
            accuracy_on_train_data=accuracy_on_train_data.item() if accuracy_on_train_data is not None else None,
            accuracy_on_test_data=accuracy_on_test_data.item() if accuracy_on_test_data is not None else None,
        )
        return final_dist_metrics

    def monte_carlo_01_error(self, dataset: Dataset, sigma: float, num_mc_samples: int=10**4, new_noise_every: int=32):
        assert not self.training, "Model should be in eval mode."
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_mc_samples)
        # New weights are drawn for every batch, but not for every sample
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=new_noise_every, pin_memory=True)
        num_errors = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(self.device), labels.to(self.device)
                x = x.view(x.size(0), -1)
                outputs = self.forward_with_noise(x, sigma=sigma)
                _, predicted = torch.max(outputs, -1)
                num_errors += (predicted != labels).sum()
            return num_errors / num_mc_samples

    def get_max_sigma(
            self,
            dataset: Dataset,
            target_error_increase: float,
            num_mc_samples: int,
            sigma_min: float=0,
            sigma_max: float=1,
            sigma_tol:float=2**(-14),
            ) -> float:
        """Returns the maximum value of sigma within [sigma_min, sigma_max] such that the noisy accuracy is at most target_acc + acc_prec"""
        total_num_sigmas = 1 / sigma_tol  # If sigma_tol = 2^(-n) then we have chosen from 2^n possible sigmas (0 not included)
        full_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        deterministic_error = 1 - self.get_full_accuracy(dataloader=full_dataloader)
        target_error = deterministic_error + target_error_increase
        noisy_error = self.monte_carlo_01_error(dataset=dataset, sigma=sigma_max, num_mc_samples=num_mc_samples)
        if noisy_error <= target_error:
            raise ValueError(f"Error at {sigma_max=} is {noisy_error=}, which is not enough to reach {target_error=}. Increase sigma_max.")
        sigmas_tried = []
        errors = []

        while True:
            sigma_new = (sigma_max + sigma_min) / 2
            noisy_error = self.monte_carlo_01_error(dataset=dataset, sigma=sigma_new, num_mc_samples=num_mc_samples).item()
            print(f"For sigma={sigma_new} get error={noisy_error}")
            sigmas_tried.append(sigma_new)
            errors.append(noisy_error)
            if abs(sigma_max - sigma_min) < sigma_tol:
                return sigma_new, noisy_error, sigmas_tried, errors, total_num_sigmas
            if noisy_error < target_error:
                sigma_min = sigma_new
            else:
                sigma_max = sigma_new      

    def get_generalization_gap(self, train_loader, test_loader):
        full_train_01_error = 1 - self.get_full_accuracy(train_loader)
        full_test_01_error = 1 - self.get_full_accuracy(test_loader)
        return full_test_01_error - full_train_01_error

    def get_full_margin_loss(self, dataloader, margin, take_softmax=False):
        assert not self.training, "Model should be in eval mode."
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

    def get_full_kl_loss(self, base_model, domain_dataloader):
        assert not self.training, "Model should be in eval mode."
        total_dist_kl_loss = torch.tensor(0.0, device=self.device)
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        for x, _ in domain_dataloader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = F.log_softmax(self(x), dim=-1)
            targets = F.log_softmax(base_model(x), dim=-1)
            total_dist_kl_loss += kl_loss_fn(outputs, targets) * x.size(0)
        return total_dist_kl_loss / len(domain_dataloader.dataset)

    def get_full_kl_loss_with_logit_loader(self, logit_loader):
        assert not self.training, "Model should be in eval mode."
        total_dist_kl_loss = torch.tensor(0.0, device=self.device)
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        for x, targets in logit_loader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            outputs = F.log_softmax(self(x), dim=-1)
            total_dist_kl_loss += kl_loss_fn(outputs, targets) * x.size(0)
        return total_dist_kl_loss / len(logit_loader.dataset)

    def get_max_and_mean_l2_deviation(self, base_model, domain_loader):
        assert not self.training, "Model should be in eval mode."
        max_l2 = torch.tensor(0.0, device=self.device)
        for x, _ in domain_loader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            self_output = self(x)
            base_output = base_model(x)
            l2_norms = torch.linalg.vector_norm(
                self_output - base_output, ord=2, dim=-1
            )
            max_l2 = max(max_l2, l2_norms.max())
            mean_l2 = l2_norms.mean()
        return max_l2, mean_l2

    def train_model(
        self,
        base_config: BaseConfig,
        train_loss_fn,
        test_loss_fn,
        full_train_loss_fn=None,
        ) -> BaseResults:

        self.train()

        if base_config.hyperparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        elif base_config.hyperparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        elif base_config.hyperparams.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer name {base_config.hyperparams.optimizer_name}. Should be 'sgd', 'adam' or 'rmsprop'.")

        if base_config.stopping.target_full_train_loss:
            best_loss = float("inf")
            epochs_since_improvement = 0

        for epoch in range(1, base_config.stopping.max_epochs + 1):

            # Log every epoch for the first 100, every 10th until 1000, every 100th until 10000 etc.
            log_freq = 10 ** max(0, math.floor(math.log(epoch, 10)) - 1)

            if epoch % log_freq == 0:
                print(f"Epoch [{epoch}/{base_config.stopping.max_epochs}]")

            for x, labels in base_config.data.train_loader:
                assert self.training
                x, labels = x.to(self.device), labels.to(self.device)
                x = x.view(x.size(0), -1)
                outputs = self(x)
                loss = train_loss_fn(outputs, labels)
                wandb.log({base_config.records.train_loss_name + " (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_log = {"Epoch": epoch}

            ########## Evaluate model and return to training mode ##########
            self.eval()
            if base_config.stopping.target_full_train_loss:
                full_train_loss = self.get_full_loss(
                    full_train_loss_fn, base_config.data.train_loader
                )
                epoch_log[base_config.records.train_loss_name] = (
                    full_train_loss.item()
                )
                if full_train_loss < best_loss:
                    best_loss = full_train_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            if (epoch % log_freq == 0) or (epoch == base_config.stopping.max_epochs):
                if base_config.records.get_full_test_loss:
                    test_loss = self.get_full_loss(test_loss_fn, base_config.data.test_loader)
                    epoch_log[base_config.records.test_loss_name] = test_loss.item()

                if base_config.records.get_full_train_accuracy:
                    train_accuracy = self.get_full_accuracy(base_config.data.train_loader)
                    epoch_log[base_config.records.train_accuracy_name] = train_accuracy

                if base_config.records.get_full_test_accuracy:
                    test_accuracy = self.get_full_accuracy(base_config.data.test_loader)
                    epoch_log[base_config.records.test_accuracy_name] = test_accuracy
            self.train()
            ########## Evaluate model and return to training mode ##########

            # Log metrics
            wandb.log(epoch_log)

            # Test if reached target loss
            if base_config.stopping.target_full_train_loss:
                # Reached target loss
                if full_train_loss <= base_config.stopping.target_full_train_loss:
                    reached_target = True
                    lost_patience = False
                    ran_out_of_epochs = False
                    break
                # Ran out of patience
                if epochs_since_improvement >= base_config.stopping.patience:
                    reached_target = False
                    lost_patience = True
                    ran_out_of_epochs = False
                    break
        
        # Either no target set or ran out of epochs
        else:
            # Ran out of epochs
            if base_config.stopping.target_full_train_loss:
                reached_target = False
                lost_patience = False
                ran_out_of_epochs = True
            # No target set
            else:
                full_train_loss = None
                reached_target = None
                lost_patience = None
                ran_out_of_epochs = None

        return self.get_final_base_metrics(
            base_config=base_config,
            train_loader=base_config.data.train_loader,
            test_loader=base_config.data.test_loader,
            loss_fn=full_train_loss_fn,
            reached_target=reached_target,
            epochs_taken=epoch,
            lost_patience=lost_patience,
            ran_out_of_epochs=ran_out_of_epochs,
            train_loss=full_train_loss,
            )

    def get_dist_loss_fn(self, objective, reduction, k: Optional[int], alpha: Optional[float]):

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
        base_model: MLP,
        dist_config: DistConfig,
        epoch_shift=0,
        ) -> DistAttemptResults:
        # N.B. self is interpreted as the dist model for this method
        
        # Set up loss function, optimizer and scheduler
        assert not base_model.training, "Base model should be in eval mode."
        train_loss_fn = self.get_dist_loss_fn(
            dist_config.objective.objective_name,
            dist_config.objective.reduction,
            dist_config.objective.k,
            dist_config.objective.alpha,
        )
        # train_loss_name = f"Dist Train ({dist_config.objective.objective_name} {dist_config.objective.reduction})"
        # train_loss_name = dist_config.objective.full_objective_name + " (batch)"
        train_loss_name = f"Dist Train {dist_config.objective.full_objective_name}"
        test_loss_name = f"Dist Test {dist_config.objective.full_objective_name}"
        train_acc_name = f"Dist Train Accuracy"
        test_acc_name = f"Dist Test Accuracy"
        optimizer = torch.optim.Adam(self.parameters(), lr=dist_config.hyperparams.lr)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3)
            if dist_config.objective.use_scheduler
            else None
        )

        # Initialize variables for early stopping
        if dist_config.stopping.target_kl_on_train:
            best_kl = float("inf")
            epochs_since_improvement = 0

        # Distill model
        for epoch in range(1, dist_config.stopping.max_epochs + 1):

            if epoch % dist_config.stopping.print_every == 1:
                print(f"Epoch [{epoch}/{dist_config.stopping.max_epochs}]")

            for x, targets in dist_config.data.logit_train_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                outputs = F.log_softmax(self(x), dim=-1)
                loss = train_loss_fn(outputs, targets)
                # wandb.log({train_loss_name: loss.item()})
                wandb.log({train_loss_name + " (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # alpha *= 1.5
            # alpha = min(alpha, 10**5)

            epoch_log = {"Epoch": epoch + epoch_shift}
            if dist_config.objective.objective_name == "l2":
                epoch_log["Alpha"] = dist_config.objective.alpha
            if scheduler:
                epoch_log["lr"] = scheduler.get_last_lr()[0]

            ########## Evaluate model ##########
            if dist_config.records.get_full_kl_on_train_data:
                if dist_config.data.use_whole_dataset:
                    full_kl_on_train_data = loss
                else:
                    full_kl_on_train_data = (
                        self.get_full_kl_loss_with_logit_loader(dist_config.data.logit_train_loader)
                    )
                epoch_log[dist_config.records.train_kl_name] = full_kl_on_train_data

                if full_kl_on_train_data < best_kl:
                    best_kl = full_kl_on_train_data
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            if dist_config.records.get_full_kl_on_test_data:
                full_kl_on_test_data = self.get_full_kl_loss_with_logit_loader(
                    dist_config.data.logit_test_loader
                )
                epoch_log[dist_config.records.test_kl_name] = full_kl_on_test_data

            if dist_config.records.get_full_accuracy_on_train_data:
                train_accuracy = self.get_full_accuracy(dist_config.data.domain_train_loader)
                epoch_log[dist_config.records.train_accuracy_name] = train_accuracy

            if dist_config.records.get_full_accuracy_on_test_data:
                test_accuracy = self.get_full_accuracy(dist_config.data.domain_test_loader)
                epoch_log[dist_config.records.test_accuracy_name] = test_accuracy

            # l2 deviation on test data
            # if dist_config.records.get_full_l2_on_test_data:
            #     max_l2_dev, mean_l2_dev = self.get_max_and_mean_l2_deviation(
            #         base_model, dist_config.data.domain_test_loader
            #     )
            #     epoch_log[
            #         f"Max l2 Deviation ({dist_config.objective.objective_name} {dist_config.objective.reduction})"
            #     ] = max_l2_dev
            #     epoch_log[
            #         f"Mean l2 Deviation ({dist_config.objective.objective_name} {dist_config.objective.reduction})"
            #     ] = mean_l2_dev
            ########## Evaluate model ##########

            # scheduler.step(max_l2_dev)

            wandb.log(epoch_log)

            # TODO: If successful, you should log one last time
            if dist_config.stopping.target_kl_on_train:
                if full_kl_on_train_data <= dist_config.stopping.target_kl_on_train:
                    reached_target = True
                    lost_patience = False
                    ran_out_of_epochs = False
                    break
                if epochs_since_improvement >= dist_config.stopping.patience:
                    reached_target = False
                    lost_patience = True
                    ran_out_of_epochs = False
                    break

        else:
            reached_target = False
            lost_patience = False
            ran_out_of_epochs = True

        return DistAttemptResults(
            mean_kl_on_train_data=full_kl_on_train_data,
            reached_target=reached_target,
            epochs_taken=epoch,
            lost_patience=lost_patience,
            ran_out_of_epochs=ran_out_of_epochs,
        )

    def dist_best_of_n(
            self,
            dist_config: DistConfig,
            dist_dims: list[int],
            ) -> tuple[bool, Optional[MLP]]:
            # N.B. self is interpreted as the base model for this method
        dist_model = MLP(
            dimensions=dist_dims,
            activation=dist_config.hyperparams.activation,
            device=self.device,
            shift_logits=dist_config.objective.shift_logits,
        )

        for attempt in range(dist_config.stopping.num_attempts):

            dist_trial_results = dist_model.dist_from(
                base_model=self,
                dist_config=dist_config,
                epoch_shift=attempt * dist_config.stopping.max_epochs,
                )

            if dist_trial_results.reached_target:
                return True, dist_model

            dist_model.reinitialize_weights()

        return False, None

    # def get_dist_dims(self, dim_skip):

    #     input_dim, hidden_dims, output_dim = (
    #         self.dimensions[0],
    #         self.dimensions[1:-1],
    #         self.dimensions[-1],
    #     )

    #     dist_hidden_dims = product(
    #         *[list(range(1, dim + 1, dim_skip)) for dim in hidden_dims]
    #     )
    #     dist_dims = [
    #         [input_dim] + list(h_dims) + [output_dim] for h_dims in dist_hidden_dims
    #     ]
    #     return dist_dims

    def get_logits_dataloaders(self, domain_train_loader, domain_test_loader, batch_size, use_whole_dataset, device):
        # N.B. self is interpreted as the base model for this method
        logit_train_loader = get_logits_dataloader(
            model=self,
            data_loader=domain_train_loader,
            batch_size=batch_size,
            use_whole_dataset=use_whole_dataset,
            device=device
        )
        logit_test_loader = get_logits_dataloader(
            model=self,
            data_loader=domain_test_loader,
            batch_size=batch_size,
            use_whole_dataset=use_whole_dataset,
            device=device
        )
        return logit_train_loader, logit_test_loader

    def get_dist_complexity(
            self,
            dist_config: DistConfig,
            ) -> tuple[Optional[MLP], DistFinalResults]:
        # N.B. self is interpreted as the base model for this method

        hidden_dim_guess = dist_config.hyperparams.initial_guess_hidden_dim

        print(f"Hidden dim guess: {hidden_dim_guess}")
        dist_dims = [self.dimensions[0], hidden_dim_guess, self.dimensions[-1]]
        dist_successful, dist_model = self.dist_best_of_n(
            dist_config=dist_config,
            dist_dims=dist_dims,
        )

        # If the first guess worked, halve until it doesn't
        if dist_successful:
            while dist_successful:
                # If the last one was 1 and was successful, we can stop here
                if hidden_dim_guess == 1:
                    complexity = 1
                    dist_model = dist_model_high
                    final_dist_metrics = self.get_final_dist_metrics(
                        dist_config=dist_config,
                        complexity=complexity,
                    )
                    return dist_model, final_dist_metrics
                dist_model_high = dist_model
                hidden_dim_guess //= 2
                print(f"Hidden dim guess: {hidden_dim_guess}")
                dist_dims = [self.dimensions[0], hidden_dim_guess, self.dimensions[-1]]
                dist_successful, dist_model = self.dist_best_of_n(
                    dist_config=dist_config,
                    dist_dims=dist_dims,
                )
            hidden_dim_low, hidden_dim_high = hidden_dim_guess, hidden_dim_guess * 2
            dist_model_low = dist_model

        # If the first guess didn't work, double until it does
        else:
            while not dist_successful:
                if hidden_dim_guess > dist_config.hyperparams.max_hidden_dim:
                    raise ValueError(
                        f"Complexity (current guess: {hidden_dim_guess}) is larger than max hidden dim {dist_config.hyperparams.max_hidden_dim}."
                    )
                dist_model_low = dist_model
                hidden_dim_guess *= 2
                print(f"Hidden dim guess: {hidden_dim_guess}")
                dist_dims = [self.dimensions[0], hidden_dim_guess, self.dimensions[-1]]
                dist_successful, dist_model = self.dist_best_of_n(
                    dist_config=dist_config,
                    dist_dims=dist_dims,
                )
            hidden_dim_low, hidden_dim_high = hidden_dim_guess // 2, hidden_dim_guess * 2
            dist_model_high = dist_model

        # Start the binary search
        while hidden_dim_high - hidden_dim_low > 1:

            print(f"Hidden dim range: ({hidden_dim_low}, {hidden_dim_high})")
            hidden_dim_mid = (hidden_dim_low + hidden_dim_high) // 2
            dist_dims = [self.dimensions[0], hidden_dim_mid, self.dimensions[-1]]
            dist_successful, dist_model = self.dist_best_of_n(
                dist_config=dist_config,
                dist_dims=dist_dims,
            )

            if dist_successful:
                hidden_dim_high = hidden_dim_mid
                dist_model_high = dist_model
            else:
                hidden_dim_low = hidden_dim_mid
                dist_model_low = dist_model

        complexity = hidden_dim_high
        dist_model = dist_model_high
        final_dist_metrics = self.get_final_dist_metrics(
            dist_config=dist_config,
            complexity=complexity,
        )

        return dist_model, final_dist_metrics

    # TODO: Needs fixing
    def get_dist_variance(
        self,
        dist_config,
        domain_train_loader,
        hidden_dim,
        num_repeats,
    ):

        dist_dims = [self.dimensions[0], hidden_dim, self.dimensions[-1]]
        print(f"Attempting to distill into model with dims {dist_dims}")
        dist_model = MLP(
            dimensions=dist_dims,
            activation=dist_config.dist_activation,
            device=self.device,
            shift_logits=dist_config.shift_logits,
        )

        kl_losses_and_epochs = []
        for rep in range(num_repeats):

            total_kl_loss_on_train_data, target_loss_achieved, epochs_taken = (
                dist_model.dist_from(
                    base_model=self,
                    dist_config=dist_config,
                    epoch_shift=rep * dist_config.max_epochs,
                )
            )
            kl_losses_and_epochs.append((total_kl_loss_on_train_data, epochs_taken))
            dist_model.reinitialize_weights()

        return kl_losses_and_epochs

    def save(self, model_dir, model_name):
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_name}"
        torch.save(self.state_dict(), model_path)

    def load(self, model_dir, model_name):
        model_path = f"{model_dir}/{model_name}"
        self.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

    def max_deviation(self, base_model, epsilon, data_shape):
        mesh, actual_epsilon, actual_cell_width = get_epsilon_mesh(
            epsilon, data_shape, device=self.device
        )
        mesh = (mesh - 0.5) / 0.5
        self_output = self(mesh)
        base_output = base_model(mesh)
        l2_norms = torch.linalg.vector_norm(self_output - base_output, ord=2, dim=-1)
        return l2_norms.max(), actual_epsilon, actual_cell_width

    def get_spectral_bound(self: MLP, other: MLP, C) -> torch.Tensor:
        if not self.same_architecture(other):
            raise ValueError("Models have different architectures.")
        weight_distances = self.get_weight_distances(other)
        return self.beta(other=other, i=self.dimensions - 1, C=C, weight_distances=weight_distances)

    def alpha(self: MLP, other: MLP, i: int, C):
        if i < 0:
            raise ValueError(f"i must be positive but received {i=}")
        elif i == 0:
            return C
        else:
            return other.weight_norms(layer_num=i) * self.alpha(other=other, i=i - 1, C=C) + self.bias_norms(layer_num=i)

    def beta(self: MLP, other: MLP, i: int, C, weight_distances):
        if i < 0:
            raise ValueError(f"i must be positive but received {i=}")
        elif i == 0:
            return 0
        else:
            return self.weight_norms(layer_num=i) * self.beta(other=other, i=i - 1, C=C) + weight_distances[i] * self.alpha(other=other, i=i - 1, C=C)

    def weight_norms(self, layer_num):
        if self._weight_norms is None:
            self._weight_norms = {i + 1: torch.linalg.norm(layer.weight.detach(), ord=2) for i, layer in enumerate(self.layers)}
        return self._weight_norms[layer_num]

    def bias_norms(self, layer_num):
        if self._bias_norms is None:
            self._bias_norms = {i + 1: torch.linalg.norm(layer.bias.detach(), ord=2) for i, layer in enumerate(self.layers)}
        return self._bias_norms[layer_num]

    def get_weight_distances(self: MLP, other: MLP):
        return {i + 1: torch.linalg.norm(layer_self.weight.detach() - layer_other.weight.detach(), ord=2) for i, (layer_self, layer_other) in enumerate(zip(self.layers, other.layers))}

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

    def get_concatenated_weights(self):
        """Returns the concatenated weights of the model as a single (num_weights, 1) tensor"""
        weights = [layer.weight.detach().clone().view(-1, 1) for layer in self.layers]
        return torch.cat(weights, dim=0)

    @torch.no_grad()
    def set_from_concatenated_weights(self, weights):
        """Sets the weights of the model from a concatenated (num_weights, 1) tensor"""
        if len(weights) != self.num_weights:
            raise ValueError(f"MLP has {self.num_weights} weights, but got {len(weights)} weights.")
        i = 0
        for layer in self.layers:
            num_weights = layer.weight.numel()
            layer.weight.copy_(weights[i:i+num_weights].view(layer.weight.shape))
            i += num_weights

    def get_quantized_model(self, codeword_length: int) -> MLP:
        """Returns a new, quantized model by applying k-means clustering to the weights"""
        if codeword_length not in range(1, 33):
            raise ValueError(f"Codeword length must be in range [1, ..., 32] but received {codeword_length=}")
        if codeword_length == 32:
            quantized_model = deepcopy(self)
        else:
            num_codewords = 2 ** codeword_length
            concatenated_weights = self.get_concatenated_weights()
            kmeans = KMeans(n_clusters=num_codewords, random_state=0).fit(concatenated_weights.cpu().numpy())
            quantized_weights = kmeans.cluster_centers_[kmeans.labels_]
            quantized_model = deepcopy(self)
            quantized_model.set_from_concatenated_weights(torch.tensor(quantized_weights, device=self.device))
        quantized_model.eval()
        return quantized_model

    def quantized_size_in_bits(self, codeword_length: int) -> int:
        """"Returns the number of bits required to specify the quantized model, including the codebook"""
        weights_size = self.num_weights * codeword_length
        biases_size = self.num_biases * 32
        codebook_size = 2 ** codeword_length * 32
        return weights_size + biases_size + codebook_size

    def get_KL_of_quantized_model(self, codeword_length: int) -> torch.Tensor:
        """We use a uniform prior over the possible codeword_lengths 1,...,32, which
        implicitly gives a uniform prior over the 32 possible quantized model bit
        lengths. The final log(32) term is -log(prior(quantized_length)), making this
        a union bound applying to the 32 possible quantized models simultaneously.
        Note there's no sigma as the posterior is a point mass on the final classifier."""
        return self.quantized_size_in_bits(codeword_length) * torch.log(torch.tensor(2)) + torch.log(torch.tensor(32))

    def get_quantized_pacb_results(
            self: MLP,
            delta: float,
            train_loader: DataLoader,
            test_loader: DataLoader,
            codeword_length: int
        ) -> QuantResults:
        """Returns the pac bound on the margin loss of the quantized model, which is
        then the bound on the error rate of the original model. The prior spreads its
        mass across different codeword lengths, so is valid for all codeword lengths
        simultaneously."""
        quant_model = self.get_quantized_model(codeword_length=codeword_length)
        spectral_bound = self.get_spectral_bound(other=quant_model)
        margin = torch.sqrt(torch.tensor(2)) * spectral_bound

        quant_train_accuracy = quant_model.get_full_accuracy(dataloader=train_loader)
        quant_test_accuracy = quant_model.get_full_accuracy(dataloader=test_loader)
        train_margin_loss = quant_model.get_full_margin_loss(dataloader=train_loader, margin=margin)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?

        quant_KL = quant_model.get_KL_of_quantized_model(codeword_length=codeword_length)
        quant_kl_bound = pacb_kl_bound(KL=quant_KL, n=len(train_loader.dataset), delta=delta)
        quant_error_bound_inverse_kl = pacb_error_bound_inverse_kl(empirical_error=train_margin_loss, KL=quant_KL, n=len(train_loader.dataset), delta=delta)
        quant_error_bound_pinsker = pacb_error_bound_pinsker(empirical_error=train_margin_loss, KL=quant_KL, n=len(train_loader.dataset), delta=delta)

        quant_results = QuantResults(
            codeword_length=codeword_length,
            spectral_bound=spectral_bound,
            margin=margin,
            train_accuracy=quant_train_accuracy,
            test_accuracy=quant_test_accuracy,
            train_margin_loss=train_margin_loss,
            KL=quant_KL,
            kl_bound=quant_kl_bound,
            error_bound_inverse_kl=quant_error_bound_inverse_kl,
            error_bound_pinsker=quant_error_bound_pinsker,
        )
        return quant_results

# TODO: Implement the KL divergence
class LowRankMLP(MLP):
    def __init__(self, dimensions, activation, device="cpu"):
        super().__init__(
            dimensions=dimensions,
            activation=activation,
            low_rank=True,
            device=device,
        )
        self.d = len(dimensions) - 1

        self._weight_norms = None
        self._bias_norms = None
        self._low_rank_weight_norms = None
        self._perturbation_norms = None

        self.rank_combs = list(
            product(*[range(1, min(layer.weight.shape) + 1) for layer in self.layers])
        )
        # self._product_weight_spectral_norms = None
        # self._valid_rank_combs = None
        # self._epsilons = None  # Note this is without the B from Lemma 2 in the paper
        self._num_params = None
        self._min_UVs = None
        self._max_UVs = None
        self._min_Ss = None
        self._max_Ss = None
        self._KL_divergences = None

    def forward(self, x: torch.Tensor, ranks: list[int]):
        if len(ranks) != self.d:
            raise ValueError(f"Expected {self.d} ranks, got {ranks=}.")
        i = 0
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                x = layer(x, rank=ranks[i])
                i += 1
            else:
                x = layer(x)
        return x

    def weight_norms(self, layer_num):
        if self._weight_norms is None:
            self._weight_norms = {i + 1: layer.weight_norm for i, layer in enumerate(self.layers)}
        return self._weight_norms[layer_num]

    def bias_norms(self, layer_num):
        if self._bias_norms is None:
            self._bias_norms = {i + 1: layer.bias_norm for i, layer in enumerate(self.layers)}
        return self._bias_norms[layer_num]

    def low_rank_weight_norms(self, layer_num, rank):    
        if self._weight_norms is None:
            self._weight_norms = dict()
            for l_num in range(1, self.d + 1):
                for r in range(1, min(self.layers[l_num].weight.shape) + 1):
                    self._weight_norms[(l_num, r)] = self.layers[l_num].low_rank_Ws_spectral_norms[r]
        return self._weight_norms[(layer_num, rank)]

    def perturbation_norms(self, layer_num, rank):
        if self._perturbation_norms is None:
            self._perturbation_norms = dict()
            for l_num in range(1, self.d + 1):
                for r in range(1, min(self.layers[l_num].weight.shape) + 1):
                    self._perturbation_norms[(l_num, r)] = self.layers[l_num]._perturbation_norms[r]
        return self._perturbation_norms[(layer_num, rank)]

    # def alpha(self, C, i: int):
    #     if i == 0:
    #         return C
    #     else:
    #         return self.weight_norms(layer_num=i) * self.alpha(C, i - 1) + self.bias_norms(layer_num=i)

    # def beta(self, C, i: int, ranks: list[int]):
    #     if i == 1:
    #         return self.perturbation_norms(layer_num=i, rank=ranks[0]) * C
    #     else:
    #         return self.low_rank_weight_norms(layer_num=i, rank=ranks[i - 1]) * self.beta(C, i - 1, ranks=ranks) + self.perturbation_norms(layer_num=i, rank=ranks[i - 1]) * self.alpha(C, i - 1)

    def max_l2_deviation(self, C, ranks: list[int]):
        return self.beta(C, i=self.d, ranks=ranks)

    def KL_divergences(self):
        pass

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

    # @property
    # def valid_rank_combs(self):
    #     """Returns a dictionary with rank combinations as keys and booleans as values,
    #     indicating whether the rank combination is valid produces a perturbation U within
    #     the tolerance of Lemma 2 from Neyshabur et al. 2017 (spectrally-normalized margin bounds)."""
    #     if self._valid_rank_combs is None:
    #         self._valid_rank_combs = {}
    #         for rank_comb in self.rank_combs:
    #             self._valid_rank_combs[rank_comb] = all(
    #                 [
    #                     layer.perturbation_spectral_norms[rank]
    #                     <= layer.weight_spectral_norm / len(self.layers)
    #                     for rank, layer in zip(rank_comb, self.layers)
    #                 ]
    #             )
    #     return self._valid_rank_combs

    # @property
    # def epsilons(self):
    #     if self._epsilons is None:
    #         self._epsilons = {}
    #         for rank_comb in self.rank_combs:
    #             norm_ratios = [
    #                 layer.perturbation_spectral_norms[rank] / layer.weight_spectral_norm
    #                 for rank, layer in zip(rank_comb, self.layers)
    #             ]
    #             self._epsilons[rank_comb] = (
    #                 torch.e * self.product_weight_spectral_norms * sum(norm_ratios)
    #             )
    #     return self._epsilons


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

    def get_hyper_model_scaled_input(self, hyper_config):
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

    def get_hyper_model_binary_input(self, hyper_config):
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
    return base_model_estimate.get_full_accuracy(dataloader).item()
