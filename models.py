from __future__ import annotations
import os
from typing import Optional, Union
from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import math
import wandb
from sklearn.cluster import KMeans

from config import BaseConfig, BaseResults, DistConfig, DistAttemptResults, DistFinalResults, CompResults
from kl_utils import distillation_loss, kl_scalars_inverse, pacb_kl_bound, pacb_error_bound_inverse_kl, pacb_error_bound_pinsker
from truncation_utils import truncate


class MLP(nn.Module):
    def __init__(
        self,
        dimensions,
        activation_name,
        dropout_prob=0.0,
        device="cpu",
        shift_logits=False
    ):
        super(MLP, self).__init__()
        self.dimensions = dimensions
        self.activation_name = activation_name
        self.activation_func = self.get_act(activation_name)
        self.dropout_prob = dropout_prob
        self.network_modules = []
        self.device = torch.device(device)
        self.shift_logits = shift_logits

        for i in range(len(dimensions) - 1):
            self.network_modules.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:  # No activation or dropout on the last layer
                self.network_modules.append(self.activation_func())
                if self.dropout_prob > 0:
                    self.network_modules.append(nn.Dropout(p=self.dropout_prob))

        self.network = nn.Sequential(*self.network_modules).to(self.device)
        self.linear_layers = [layer for layer in self.network_modules if isinstance(layer, nn.Linear)]
        self.num_parameters = sum([p.numel() for p in self.parameters()])
        self.num_weights = sum([layer.weight.numel() for layer in self.linear_layers])
        self.num_biases = sum([layer.bias.numel() for layer in self.linear_layers])

        # SVD Decompositions. Note this does not make the model low rank; the Us, Ss and Vs are just used by get_low_rank_model()
        self.Us = None
        self.Ss = None
        self.Vts = None
        self.svds_computed = False

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
        if type(self.activation_func) != type(other.activation_func):
            return False
        return True

    def square_l2_dist(self: MLP, other: MLP) -> torch.Tensor:
        with torch.no_grad():
            total = 0
            for layer_self, layer_other in zip(self.linear_layers, other.linear_layers):
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
        for layer in self.linear_layers:
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
        kl_on_train_data = self.get_full_kl_loss_with_logit_loader(dist_config.data.base_logit_train_loader) if dist_config.records.get_final_kl_on_train_data else None
        kl_on_test_data = self.get_full_kl_loss_with_logit_loader(dist_config.data.base_logit_test_loader) if dist_config.records.get_final_kl_on_test_data else None
        accuracy_on_train_data = self.get_full_accuracy(dist_config.data.domain_train_loader) if dist_config.records.get_full_accuracy_on_train_data else None
        accuracy_on_test_data = self.get_full_accuracy(dist_config.data.domain_test_loader) if dist_config.records.get_full_accuracy_on_test_data else None
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

    def get_full_kl_loss(self, base_model: MLP, domain_dataloader: DataLoader):
        assert not self.training, "Model should be in eval mode."
        total_dist_kl_loss = torch.tensor(0.0, device=self.device)
        for x, _ in domain_dataloader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            student_log_probs = F.log_softmax(self(x), dim=-1)
            student_probs = torch.exp(student_log_probs)
            teacher_log_probs = F.log_softmax(base_model(x), dim=-1)
            total_dist_kl_loss += distillation_loss(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
            ) * x.size(0)
        return total_dist_kl_loss / len(domain_dataloader.dataset)

    def get_full_kl_loss_with_logit_loader(self, logit_loader: DataLoader):
        assert not self.training, "Model should be in eval mode."
        total_dist_kl_loss = torch.tensor(0.0, device=self.device)
        for x, targets, base_logits, base_log_probs, base_probs in logit_loader:
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            student_log_probs = F.log_softmax(self(x), dim=-1)
            student_probs = torch.exp(student_log_probs)
            total_dist_kl_loss += distillation_loss(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=base_log_probs
            ) * x.size(0)
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

        print(f"Training on {len(base_config.data.train_loader.dataset)} samples from {base_config.data.dataset_name}")

        if base_config.hyperparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        elif base_config.hyperparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        elif base_config.hyperparams.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=base_config.hyperparams.lr, weight_decay=base_config.hyperparams.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer name {base_config.hyperparams.optimizer_name}. Should be 'sgd', 'adam' or 'rmsprop'.")

        # Initialize early stopping variables
        best_loss = float("inf")
        epochs_since_improvement = 0
        reached_target = False
        lost_patience = False
        ran_out_of_epochs = False

        for epoch in range(1, base_config.stopping.max_epochs + 1):
            
            self.train()

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

            # Evaluate model
            self.eval()
            full_train_loss = None

            if (epoch % log_freq == 0) or (epoch == base_config.stopping.max_epochs):
                if base_config.records.get_full_train_loss:
                    full_train_loss = self.get_full_loss(full_train_loss_fn, base_config.data.train_loader)
                    epoch_log[base_config.records.train_loss_name] = full_train_loss.item()

                if base_config.records.get_full_test_loss:
                    test_loss = self.get_full_loss(test_loss_fn, base_config.data.test_loader)
                    epoch_log[base_config.records.test_loss_name] = test_loss.item()

                if base_config.records.get_full_train_accuracy:
                    train_accuracy = self.get_full_accuracy(base_config.data.train_loader)
                    epoch_log[base_config.records.train_accuracy_name] = train_accuracy

                if base_config.records.get_full_test_accuracy:
                    test_accuracy = self.get_full_accuracy(base_config.data.test_loader)
                    epoch_log[base_config.records.test_accuracy_name] = test_accuracy

            # Log metrics

            # Test if reached target loss
            if base_config.stopping.use_early_stopping:

                # full_train_loss may not have been calculated earlier
                if full_train_loss is None:
                    full_train_loss = self.get_full_loss(full_train_loss_fn, base_config.data.train_loader)
                    epoch_log[base_config.records.train_loss_name] = full_train_loss.item()

                # Check if target loss reached
                if base_config.stopping.target_full_train_loss:
                    if full_train_loss <= base_config.stopping.target_full_train_loss:
                        reached_target = True
                        print(f"Target loss reached at epoch {epoch}.")
                        wandb.log(epoch_log)
                        break

                # Check patience criterion
                if base_config.stopping.patience is not None:
                    if full_train_loss < best_loss:
                        best_loss = full_train_loss
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

                    # Ran out of patience
                    if epochs_since_improvement >= base_config.stopping.patience:
                        lost_patience = True
                        print(f"Ran out of patience at epoch {epoch}.")
                        wandb.log(epoch_log)
                        break

            wandb.log(epoch_log)
        
        else:
            ran_out_of_epochs = True
            print(f"Ran out of epochs at epoch {epoch}.")
            if not base_config.stopping.use_early_stopping:
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
        
        # Set up loss function and optimizer
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

        # Initialize variables for early stopping
        if dist_config.stopping.target_kl_on_train:
            best_kl = float("inf")
            epochs_since_improvement = 0

        # Distill model
        for epoch in range(1, dist_config.stopping.max_epochs + 1):

            if epoch % dist_config.stopping.print_every == 1:
                print(f"Epoch [{epoch}/{dist_config.stopping.max_epochs}]")

            for x, targets, base_logits, base_log_probs, base_probs in dist_config.data.base_logit_train_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                student_log_probs = F.log_softmax(self(x), dim=-1)
                student_probs = torch.exp(student_log_probs)
                loss = distillation_loss(
                    student_probs=student_probs,
                    student_log_probs=student_log_probs,
                    teacher_log_probs=base_log_probs,
                )

                # loss = train_loss_fn(dist_logits, base_logits)
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

            ########## Evaluate model ##########
            if dist_config.records.get_full_kl_on_train_data:
                if dist_config.data.use_whole_dataset:
                    full_kl_on_train_data = loss
                else:
                    full_kl_on_train_data = (
                        self.get_full_kl_loss_with_logit_loader(dist_config.data.base_logit_train_loader)
                    )
                epoch_log[dist_config.records.train_kl_name] = full_kl_on_train_data

                if full_kl_on_train_data < best_kl:
                    best_kl = full_kl_on_train_data
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            if dist_config.records.get_full_kl_on_test_data:
                full_kl_on_test_data = self.get_full_kl_loss_with_logit_loader(
                    dist_config.data.base_logit_test_loader
                )
                epoch_log[dist_config.records.test_kl_name] = full_kl_on_test_data

            if dist_config.records.get_full_accuracy_on_train_data:
                train_accuracy = self.get_full_accuracy(dist_config.data.domain_train_loader)
                epoch_log[dist_config.records.train_accuracy_name] = train_accuracy

            if dist_config.records.get_full_accuracy_on_test_data:
                test_accuracy = self.get_full_accuracy(dist_config.data.domain_test_loader)
                epoch_log[dist_config.records.test_accuracy_name] = test_accuracy
            ########## Evaluate model ##########

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
            activation_name=dist_config.hyperparams.activation_name,
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
            activation_name=dist_config.dist_activation,
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

    @torch.no_grad()
    def get_empirical_l2_bound(self: MLP, other: MLP, dataloader: DataLoader, base_logit_loader: Optional[DataLoader] = None) -> torch.Tensor:
        # N.B. self is interpreted as the base model for this method, and logit_loader should yield logits from the base model
        max_empirical_l2 = torch.tensor(0.0, device=self.device)
        if base_logit_loader is not None:
            for x, self_targets, self_logits, self_log_probs, self_probs in base_logit_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                self_logits = self_logits
                other_logits = other(x)
                l2_norms = torch.linalg.vector_norm(self_logits - other_logits, ord=2, dim=-1)
                max_empirical_l2 = max(max_empirical_l2, l2_norms.max())
            return max_empirical_l2
        else:
            for x, _ in dataloader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                self_logits = self(x)
                other_logits = other(x)
                l2_norms = torch.linalg.vector_norm(self_logits - other_logits, ord=2, dim=-1)
                max_empirical_l2 = max(max_empirical_l2, l2_norms.max())
            return max_empirical_l2

    def get_spectral_l2_bound(self: MLP, other: MLP, C) -> torch.Tensor:
        if not self.same_architecture(other):
            raise ValueError("Models have different architectures.")
        return self.beta(other=other, layer_num=len(self.dimensions) - 1, C=C)

    def alpha(self: MLP, other: MLP, layer_num: int, C):
        if layer_num < 0:
            raise ValueError(f"layer_num must be positive but received {layer_num=}")
        elif layer_num == 0:
            return C
        else:
            return other.weight_norm(layer_num=layer_num) * self.alpha(other=other, layer_num=layer_num - 1, C=C) + self.bias_norm(layer_num=layer_num)

    def beta(self: MLP, other: MLP, layer_num: int, C):
        if layer_num < 0:
            raise ValueError(f"layer_num must be positive but received {layer_num=}")
        elif layer_num == 0:
            return 0
        else:
            return self.weight_norm(layer_num=layer_num) * self.beta(other=other, layer_num=layer_num - 1, C=C) + self.weight_distance(other, layer_num=layer_num) * self.alpha(other=other, layer_num=layer_num - 1, C=C)

    @torch.no_grad()
    def weight_norm(self, layer_num: int) -> torch.Tensor:
        layer_idx = layer_num - 1
        layer = self.linear_layers[layer_idx]
        return torch.linalg.norm(layer.weight, ord=2)

    @torch.no_grad()
    def bias_norm(self, layer_num: int) -> torch.Tensor:
        layer_idx = layer_num - 1
        layer = self.linear_layers[layer_idx]
        return torch.linalg.norm(layer.bias, ord=2)

    @torch.no_grad()
    def weight_distance(self: MLP, other: MLP, layer_num: int) -> torch.Tensor:
        layer_idx = layer_num - 1
        layer_self = self.linear_layers[layer_idx]
        layer_other = other.linear_layers[layer_idx]
        return torch.linalg.norm(layer_self.weight - layer_other.weight, ord=2)

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

    def get_concatenated_weights(self) -> torch.Tensor:
        """Returns the concatenated weights of the model as a single (num_weights, 1) tensor"""
        weights = [layer.weight.detach().clone().view(-1, 1) for layer in self.linear_layers]
        return torch.cat(weights, dim=0)

    @torch.no_grad()
    def set_from_concatenated_weights(self, concatenated_weights: torch.Tensor) -> None:
        """Sets the weights of the model from a concatenated (num_weights, 1) tensor"""
        if len(concatenated_weights) != self.num_weights:
            raise ValueError(f"MLP has {self.num_weights} weights, but got {len(concatenated_weights)} weights.")
        i = 0
        for layer in self.linear_layers:
            num_weights = layer.weight.numel()
            layer.weight.copy_(concatenated_weights[i:i+num_weights].view(layer.weight.shape))
            i += num_weights

    @staticmethod
    def check_comp_arguments(
            codeword_length: Optional[int],
            exponent_bits: Optional[int],
            mantissa_bits: Optional[int],
        ) -> None:
        if (exponent_bits is not None) and (mantissa_bits is not None):
            trunc = True
        elif (exponent_bits is None) and (mantissa_bits is None):
            trunc = False
        else:
            raise ValueError(f"Both {exponent_bits=} and {mantissa_bits=} must be None or both must be set.")
        if codeword_length is not None and trunc:
            raise ValueError(f"Cannot both quantize with {codeword_length=} and truncate with {exponent_bits=}, {mantissa_bits=}")

    def get_comp_model(
            self,
            ranks: Optional[tuple[int]],
            codeword_length: Optional[int],
            exponent_bits: Optional[int],
            mantissa_bits: Optional[int],
        ) -> Union[MLP, LowRankMLP]:
        """Returns a compressed model. If ranks is not None, returns a low-rank model."""
        # Check arguments
        self.check_comp_arguments(
            codeword_length=codeword_length,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
        )
        quant = codeword_length is not None
        trunc = exponent_bits is not None and mantissa_bits is not None
        # Construct compressed model
        if ranks is not None:
            comp_model = self.get_low_rank_model(ranks=ranks)
        else:
            comp_model = deepcopy(self)
        if quant:
            comp_model = comp_model.get_quant_k_means_model(codeword_length=codeword_length)
        if trunc:
            comp_model = comp_model.get_quant_trunc_model(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)
        comp_model.eval()
        return comp_model

    # TODO: Should this go somewhere else?
    def get_num_UV_truncs(self, ranks: tuple[int]) -> int:
        """Returns the number of U_truncs and V_truncs in the low-rank model."""
        num_UV_truncs = 0
        for i in range(1, len(self.dimensions)):
            num_UV_truncs += ranks[i-1] * (self.dimensions[i-1] + self.dimensions[i])
        return num_UV_truncs

    def get_comp_model_size_in_bits(
            self,
            ranks: Optional[tuple[int]],
            codeword_length: Optional[int],
            exponent_bits: Optional[int],
            mantissa_bits: Optional[int]
        ) -> int:
        """Returns the number of bits required to specify the compressed model, without actually creating the compressed model."""
        MLP.check_comp_arguments(
            codeword_length=codeword_length,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
        )
        low_rank = ranks is not None
        quant = codeword_length is not None
        trunc = exponent_bits is not None and mantissa_bits is not None
        if not low_rank:
            if not quant and not trunc:
                return 32 * self.num_weights + 32 * self.num_biases
            if quant and not trunc:
                return codeword_length * self.num_weights + 32 * self.num_biases + 32 * 2 ** codeword_length
            elif trunc and not quant:
                return (1 + exponent_bits + mantissa_bits) * self.num_weights + 32 * self.num_biases
            else:
                raise ValueError(f"Cannot both quantize with {codeword_length=} and truncate with {exponent_bits=}, {mantissa_bits=}")
        else:
            if not quant and not trunc:
                return 32 * self.get_num_UV_truncs(ranks=ranks) + 32 * sum(ranks) + 32 * self.num_biases
            if quant and not trunc:
                return codeword_length * self.get_num_UV_truncs(ranks=ranks) + 32 * sum(ranks) + 32 * self.num_biases + 32 * 2 ** codeword_length
            elif trunc and not quant:
                return (1 + exponent_bits + mantissa_bits) * self.get_num_UV_truncs(ranks=ranks) + 32 * sum(ranks) + 32 * self.num_biases
            else:
                raise ValueError(f"Cannot both quantize with {codeword_length=} and truncate with {exponent_bits=}, {mantissa_bits=}")

    def get_KL_of_comp_model(
            self,
            ranks: Optional[tuple[int]],
            codeword_length: Optional[int],
            exponent_bits: Optional[int],
            mantissa_bits: Optional[int]
        ) -> torch.Tensor:
        return self.get_comp_model_size_in_bits(
            ranks=ranks,
            codeword_length=codeword_length,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits
        ) + torch.log(torch.tensor(2))

    def get_low_rank_model(self: MLP, ranks: tuple[int]) -> LowRankMLP:
        """Creates a low-rank version of the MLP by truncating the SVDs of the weight matrices."""
        return LowRankMLP(self, ranks)

    def get_quant_k_means_model(self, codeword_length: int) -> MLP:
        """Returns a new, quantized model by applying k-means clustering to the weights.
        k-means takes a very long time for large values of k, for long codewords (and
        therefore large values of k) we just use the initialization of k-means."""
        if codeword_length not in range(1, 33):
            raise ValueError(f"Codeword length must be in range [1, ..., 32] but received {codeword_length=}")
        if codeword_length == 32:
            quant_k_means_model = deepcopy(self)
        else:
            num_codewords = 2 ** codeword_length
            concatenated_weights = self.get_concatenated_weights()
            kmeans = KMeans(n_clusters=num_codewords, random_state=0).fit(concatenated_weights.cpu().numpy())
            print(f"Num iterations: {kmeans.n_iter_}")
            # plt.hist(concatenated_weights.cpu().numpy(), bins=1000)
            # plt.vlines(kmeans.cluster_centers_, ymin=0, ymax=3000, color='r')
            # plt.show()
            quant_k_means_weights = kmeans.cluster_centers_[kmeans.labels_]
            quant_k_means_model = deepcopy(self)
            quant_k_means_model.set_from_concatenated_weights(torch.tensor(quant_k_means_weights, device=self.device))
        quant_k_means_model.eval()
        return quant_k_means_model

    def get_quant_trunc_model(self, exponent_bits: int, mantissa_bits: int) -> MLP:
        """Returns a new model with the weights truncated to the given number of bits."""
        quant_trunc_model = deepcopy(self)
        concatenated_weights = quant_trunc_model.get_concatenated_weights()
        quant_trunc_weights = truncate(x=concatenated_weights, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)
        quant_trunc_model.set_from_concatenated_weights(quant_trunc_weights)
        quant_trunc_model.eval()
        return quant_trunc_model

    # # TODO: Pass a ranks tuple to this. If no ranks passed, return the size of the quantized weights.
    # # If ranks passed, return the size of the quantized U_truncs and V_truncs.
    # def quant_k_means_size_in_bits(self, codeword_length: int) -> int:
    #     """"Returns the number of bits required to specify the quantized model, including the codebook"""
    #     if codeword_length == 32:
    #         num_codewords = 0  # no quantization
    #     else:
    #         num_codewords = 2 ** codeword_length
    #     # Only the weights are quantized, not the biases
    #     weights_size = self.num_weights * codeword_length
    #     biases_size = self.num_biases * 32
    #     codebook_size = num_codewords * 32
    #     return weights_size + biases_size + codebook_size

    # def get_KL_of_quant_k_means_model(self, codeword_length: int) -> torch.Tensor:
    #     """The KL between a point mass on the compressed representation of the model, with bit length
    #     b = self.quantized_size_in_bits(), against a uniform prior over the 2**b possible bit strings.
    #     This assumes that the codeword_length is fixed. A union bound should later account
    #     for any choice made over the codeword_length. Note there's no sigma as the posterior
    #     is a point mass on the final classifier."""
    #     return self.quant_k_means_size_in_bits(codeword_length=codeword_length) + torch.log(torch.tensor(2))

    # def quant_trunc_size_in_bits(self, exponent_bits: int, mantissa_bits: int) -> int:
    #     """Returns the number of bits required to specify the truncated model."""
    #     return (1 + exponent_bits + mantissa_bits) * self.num_weights + 32 * self.num_biases

    # def get_KL_of_quant_trunc_model(self, exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    #     return self.quant_trunc_size_in_bits(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits) + torch.log(torch.tensor(2))

    def get_comp_pacb_results(
            self: MLP,
            delta: float,
            num_union_bounds: int,
            train_loader: DataLoader,
            test_loader: DataLoader,
            rand_domain_loader: DataLoader,
            base_logit_train_loader: DataLoader,
            base_logit_test_loader: DataLoader,
            base_logit_rand_domain_loader: DataLoader,
            C_domain: torch.Tensor,
            C_data: torch.Tensor,

            ranks: Optional[tuple[int]],
            codeword_length: Optional[int],
            exponent_bits: Optional[int],
            mantissa_bits: Optional[int],
            compress_model_difference: bool = False,
            init_model: MLP = None,
        ) -> CompResults:
        """Returns the pac bound on the margin loss of the quantized model, which is
        then the bound on the error rate of the original model. The prior spreads its
        mass across different codeword lengths, so is valid for all codeword lengths
        simultaneously."""

        if compress_model_difference and init_model is None:
            raise ValueError("If compress_difference is True, init_model must be provided.")

        # Account for choice over ranks and codeword length
        delta /= num_union_bounds

        # diff_model is base_model - init_model if compress_difference is True, otherwise it's just self (aka difference from origin)
        if compress_model_difference:
            comp_diff_model = self.get_model_difference(other=init_model)
        else:
            comp_diff_model = deepcopy(self)

        # Compres model in one step
        comp_diff_model = comp_diff_model.get_comp_model(
            ranks=ranks,
            codeword_length=codeword_length,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
        )
        
        # Reconstruct the compressed model for evaluation by adding init_model back in if necessary
        if compress_model_difference:
            comp_model = comp_diff_model.get_model_sum(other=init_model)
        else:
            comp_model = deepcopy(comp_diff_model)
        
        # Get the KL of the compressed form (of base_model - init_model if compress_difference is True)
        diff_KL = self.get_KL_of_comp_model(
            ranks=ranks,
            codeword_length=codeword_length,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
        )

        # Get spectral and empirical l2 bounds
        print("Getting spectral and empirical l2 bounds")
        spectral_l2_bound_domain = self.get_spectral_l2_bound(other=comp_model, C=C_domain)
        spectral_l2_bound_data = self.get_spectral_l2_bound(other=comp_model, C=C_data)
        empirical_l2_bound_domain = self.get_empirical_l2_bound(other=comp_model, dataloader=rand_domain_loader, base_logit_loader=base_logit_rand_domain_loader)
        empirical_l2_bound_train_data = self.get_empirical_l2_bound(other=comp_model, dataloader=train_loader, base_logit_loader=base_logit_train_loader)
        empirical_l2_bound_test_data = self.get_empirical_l2_bound(other=comp_model, dataloader=test_loader, base_logit_loader=base_logit_test_loader)

        # Get spectral and empirical margins
        print("Getting spectral and empirical margins")
        margin_domain = torch.sqrt(torch.tensor(2)) * spectral_l2_bound_domain
        margin_data = torch.sqrt(torch.tensor(2)) * spectral_l2_bound_data
        margin_empirical_domain = torch.sqrt(torch.tensor(2)) * empirical_l2_bound_domain
        margin_empirical_train_data = torch.sqrt(torch.tensor(2)) * empirical_l2_bound_train_data
        margin_empirical_test_data = torch.sqrt(torch.tensor(2)) * empirical_l2_bound_test_data

        # Get empirical errors
        print("Getting empirical errors")
        comp_train_accuracy = comp_model.get_full_accuracy(dataloader=train_loader)
        comp_test_accuracy = comp_model.get_full_accuracy(dataloader=test_loader)

        # Get the empirical margin losses. Note these are *all* calculated on the train data, because it is only the margin they use that changes. More precisely, the bound we prove requires a margin which can be calculated in different ways, but you always measure the margin loss on the train data.
        print("Getting empirical margin losses")
        comp_train_margin_loss_domain = comp_model.get_full_margin_loss(dataloader=train_loader, margin=margin_domain)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?
        comp_train_margin_loss_data = comp_model.get_full_margin_loss(dataloader=train_loader, margin=margin_data)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?
        comp_train_margin_loss_empirical_domain = comp_model.get_full_margin_loss(dataloader=train_loader, margin=margin_empirical_domain)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?
        comp_train_margin_loss_empirical_train_data = comp_model.get_full_margin_loss(dataloader=train_loader, margin=margin_empirical_train_data)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?
        comp_train_margin_loss_empirical_test_data = comp_model.get_full_margin_loss(dataloader=train_loader, margin=margin_empirical_test_data)  # TODO: You're leaving the default argument take_softmax=False. Is this a good idea? You can actually try both ways, I think?

        # Get pacb bounds
        print("Getting PAC-B bounds")
        comp_kl_bound = pacb_kl_bound(KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_inverse_kl_domain = pacb_error_bound_inverse_kl(empirical_error=comp_train_margin_loss_domain, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_inverse_kl_data = pacb_error_bound_inverse_kl(empirical_error=comp_train_margin_loss_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_pinsker_domain = pacb_error_bound_pinsker(empirical_error=comp_train_margin_loss_domain, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_pinsker_data = pacb_error_bound_pinsker(empirical_error=comp_train_margin_loss_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_inverse_kl_empirical_domain = pacb_error_bound_inverse_kl(empirical_error=comp_train_margin_loss_empirical_domain, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_inverse_kl_empirical_train_data = pacb_error_bound_inverse_kl(empirical_error=comp_train_margin_loss_empirical_train_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_inverse_kl_empirical_test_data = pacb_error_bound_inverse_kl(empirical_error=comp_train_margin_loss_empirical_test_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_pinsker_empirical_domain = pacb_error_bound_pinsker(empirical_error=comp_train_margin_loss_empirical_domain, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_pinsker_empirical_train_data = pacb_error_bound_pinsker(empirical_error=comp_train_margin_loss_empirical_train_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)
        comp_error_bound_pinsker_empirical_test_data = pacb_error_bound_pinsker(empirical_error=comp_train_margin_loss_empirical_test_data, KL=diff_KL, n=len(train_loader.dataset), delta=delta)

        # Collect results together and return
        comp_results = CompResults(
            ranks=ranks,
            codeword_length=codeword_length,

            C_domain=C_domain.item(),
            C_data=C_data.item(),
            
            spectral_l2_bound_domain=spectral_l2_bound_domain.item(),
            spectral_l2_bound_data=spectral_l2_bound_data.item(),
            empirical_l2_bound_domain=empirical_l2_bound_domain.item(),
            empirical_l2_bound_train_data=empirical_l2_bound_train_data.item(),
            empirical_l2_bound_test_data=empirical_l2_bound_test_data.item(),
            
            margin_spectral_domain=margin_domain.item(),
            margin_spectral_data=margin_data.item(),
            margin_empirical_domain=margin_empirical_domain.item(),
            margin_empirical_train_data=margin_empirical_train_data.item(),
            margin_empirical_test_data=margin_empirical_test_data.item(),
            
            train_accuracy=comp_train_accuracy.item(),
            test_accuracy=comp_test_accuracy.item(),
            
            train_margin_loss_spectral_domain=comp_train_margin_loss_domain.item(),
            train_margin_loss_spectral_data=comp_train_margin_loss_data.item(),
            train_margin_loss_empirical_domain=comp_train_margin_loss_empirical_domain.item(),
            train_margin_loss_empirical_train_data=comp_train_margin_loss_empirical_train_data.item(),
            train_margin_loss_empirical_test_data=comp_train_margin_loss_empirical_test_data.item(),
            
            KL=diff_KL.item(),
            kl_bound=comp_kl_bound.item(),
            
            error_bound_inverse_kl_spectral_domain=comp_error_bound_inverse_kl_domain.item(),
            error_bound_inverse_kl_spectral_data=comp_error_bound_inverse_kl_data.item(),
            error_bound_inverse_kl_empirical_domain=comp_error_bound_inverse_kl_empirical_domain.item(),
            error_bound_inverse_kl_empirical_train_data=comp_error_bound_inverse_kl_empirical_train_data.item(),
            error_bound_inverse_kl_empirical_test_data=comp_error_bound_inverse_kl_empirical_test_data.item(),

            error_bound_pinsker_spectral_domain=comp_error_bound_pinsker_domain.item(),
            error_bound_pinsker_spectral_data=comp_error_bound_pinsker_data.item(),            
            error_bound_pinsker_empirical_domain=comp_error_bound_pinsker_empirical_domain.item(),
            error_bound_pinsker_empirical_train_data=comp_error_bound_pinsker_empirical_train_data.item(),
            error_bound_pinsker_empirical_test_data=comp_error_bound_pinsker_empirical_test_data.item(),
        )
        return comp_results

    def compute_svd(self):
        """Compute SVD for each layer and store the results"""
        if not self.svds_computed:
            self.Us = []
            self.Ss = []
            self.Vts = []
            with torch.no_grad():
                for layer in self.linear_layers:
                    U, S, Vt = torch.linalg.svd(layer.weight)
                    self.Us.append(U)
                    self.Ss.append(S)
                    self.Vts.append(Vt)
            self.svds_computed = True

    def U(self, layer_idx):
        self.compute_svd()
        return self.Us[layer_idx]

    def S(self, layer_idx):
        self.compute_svd()
        return self.Ss[layer_idx]

    def Vt(self, layer_idx):
        self.compute_svd()
        return self.Vts[layer_idx]

    def get_sensible_ranks(self, min_rank: int, rank_step: int) -> list[tuple[int]]:
        """Returns the rank combinations that reduce (or do not change) the storage size for every layer."""
        max_sensible_ranks = []
        for layer in self.linear_layers:
            m, n = layer.weight.shape
            max_rank = (m * n) // (m + 1 + n)
            max_sensible_ranks.append(max_rank)
        sensible_ranks = list(product(*[range(min_rank, r + 1, rank_step) for r in max_sensible_ranks]))
        return sensible_ranks

    def get_sensible_codeword_lengths(self) -> list[int]:
        full_model_size = self.get_comp_model_size_in_bits(
            ranks=None,
            codeword_length=None,
            exponent_bits=None,
            mantissa_bits=None,            
        )
        sensible_codeword_lengths = []
        for codeword_length in range(1, 33):
            comp_model_size = self.get_comp_model_size_in_bits(
                ranks=None,
                codeword_length=codeword_length,
                exponent_bits=None,
                mantissa_bits=None,
            )
            if comp_model_size < full_model_size:
                sensible_codeword_lengths.append(codeword_length)
        return sensible_codeword_lengths

    def get_sensible_ranks_and_codeword_lengths(self, min_rank: int, rank_step: int) -> list[tuple[tuple[int], int]]:
        full_model_size = self.get_comp_model_size_in_bits(
            ranks=None,
            codeword_length=None,
            exponent_bits=None,
            mantissa_bits=None,            
        )
        sensible_ranks_and_codeword_lengths = []
        sensible_ranks = self.get_sensible_ranks(min_rank=min_rank, rank_step=rank_step)
        for ranks in sensible_ranks:
            for codeword_length in range(1, 33):
                comp_model_size = self.get_comp_model_size_in_bits(
                    ranks=ranks,
                    codeword_length=codeword_length,
                    exponent_bits=None,
                    mantissa_bits=None,
                )
                if comp_model_size < full_model_size:
                    sensible_ranks_and_codeword_lengths.append((ranks, codeword_length))
        return sensible_ranks_and_codeword_lengths

    def get_model_difference(self: MLP, other: MLP) -> MLP:
        """Returns the MLP with weights and biases equal to self - other."""
        if not self.same_architecture(other):
            raise ValueError("Models have different architectures.")
        diff_model = deepcopy(self)
        for i, layer in enumerate(diff_model.linear_layers):
            layer.weight.data = layer.weight - other.linear_layers[i].weight
            layer.bias.data = layer.bias - other.linear_layers[i].bias
        diff_model.eval()
        return diff_model

    def get_model_sum(self: MLP, other: MLP) -> MLP:
        """Returns the MLP with weights and biases equal to self + other."""
        if not self.same_architecture(other):
            raise ValueError("Models have different architectures.")
        sum_model = deepcopy(self)
        for i, layer in enumerate(sum_model.linear_layers):
            layer.weight.data = layer.weight + other.linear_layers[i].weight
            layer.bias.data = layer.bias + other.linear_layers[i].bias
        sum_model.eval()
        return sum_model


class LowRankMLP(MLP):
    def __init__(self, original_mlp: MLP, ranks: tuple[int]):
        """
        Low-rank version of MLP with factorized weight matrices
        
        Args:
            original_mlp (MLP): The original MLP to create a low-rank version from
            ranks (tuple): Tuple of ranks for each linear layer factorization
        """
        super().__init__(
            dimensions=original_mlp.dimensions,
            activation_name=original_mlp.activation_name,
            dropout_prob=0.0,
            device=original_mlp.device,
            shift_logits=False,
        )
        self.eval()

        self.U_truncs = []
        self.S_truncs = []
        self.Vt_truncs = []
        self.ranks = ranks

        for i, layer in enumerate(self.linear_layers):
            U = original_mlp.U(i)
            S = original_mlp.S(i)
            Vt = original_mlp.Vt(i)
            rank = ranks[i]

            U_trunc = U[:, :rank]
            S_trunc = S[:rank]
            Vt_trunc = Vt[:rank, :]
            self.U_truncs.append(U_trunc)
            self.S_truncs.append(S_trunc)
            self.Vt_truncs.append(Vt_trunc)
            
            with torch.no_grad():
                layer.weight.data = U_trunc @ torch.diag(S_trunc) @ Vt_trunc

        self.num_UV_trunc_vals = sum([U_trunc.numel() + Vt_trunc.numel() for U_trunc, Vt_trunc in zip(self.U_truncs, self.Vt_truncs)])
        self.num_S_trunc_vals = sum([S_trunc.numel() for S_trunc in self.S_truncs])

    # def get_sensible_codeword_lengths(self) -> list[int]:
    #     codeword_lengths = []
    #     for codeword_length in range(1, 33):
    #         if self.quant_k_means_size_in_bits(codeword_length=codeword_length) <= self.quant_k_means_size_in_bits(codeword_length=32):  # Quantization should lead to a reduction in storage size
    #             if 2 ** codeword_length <= self.num_UV_trunc_vals:  # Should not have more codewords than weights
    #                 codeword_lengths.append(codeword_length)
    #     return codeword_lengths

    @torch.no_grad()
    def update_weights(self) -> None:
        for layer_idx, layer in enumerate(self.linear_layers):
            U_trunc = self.U_truncs[layer_idx]
            S_trunc = self.S_truncs[layer_idx]
            Vt_trunc = self.Vt_truncs[layer_idx]
            layer.weight.data = U_trunc @ torch.diag(S_trunc) @ Vt_trunc

    def get_concatenated_UV_truncs(self) -> torch.Tensor:
        """Returns the concatenated U_truncs and Vt_truncs of the model as a single (num_UV_trunc_vals, 1) tensor"""
        U_truncs = [U_trunc.reshape(-1, 1) for U_trunc in self.U_truncs]
        Vt_truncs = [Vt_trunc.reshape(-1, 1) for Vt_trunc in self.Vt_truncs]
        return torch.cat(U_truncs + Vt_truncs, dim=0)

    @torch.no_grad()
    def set_from_concatenated_UV_truncs(self, concatenated_UV_truncs: torch.Tensor) -> None:
        """Sets the U_truncs and V_truncs of the model from a concatenated (num_UV_trunc_vals, 1) tensor"""
        if len(concatenated_UV_truncs) != self.num_UV_trunc_vals:
            raise ValueError(f"LowRankMLP has {self.num_UV_trunc_vals} U_trunc and V_trunc vals, but got {len(concatenated_UV_truncs)}.")
        i = 0
        for U_trunc in self.U_truncs:
            num_U_vals = U_trunc.numel()
            U_trunc.copy_(concatenated_UV_truncs[i:i+num_U_vals].view(U_trunc.shape))
            i += num_U_vals
        for Vt_trunc in self.Vt_truncs:
            num_Vt_vals = Vt_trunc.numel()
            Vt_trunc.copy_(concatenated_UV_truncs[i:i+num_Vt_vals].view(Vt_trunc.shape))
            i += num_Vt_vals
        if i != len(concatenated_UV_truncs):
            raise ValueError(f"Expected {len(concatenated_UV_truncs)} values but got {i}.")
    
    def get_quant_k_means_model(self, codeword_length: int) -> LowRankMLP:
        """Returns a new, quantized model by applying k-means clustering to the U_truncs
        and V_truncs. Note the S_truncs are not quantized. Overwrites same named method in MLP."""
        if codeword_length not in range(1, 33):
            raise ValueError(f"Codeword length must be in range [1, ..., 32] but received {codeword_length=}")
        if codeword_length == 32:
            quant_k_means_model = deepcopy(self)
        else:
            num_codewords = 2 ** codeword_length
            concatenated_UV_truncs = self.get_concatenated_UV_truncs()
            kmeans = KMeans(n_clusters=num_codewords, random_state=0).fit(concatenated_UV_truncs.cpu().numpy())
            quant_k_means_UV_truncs = kmeans.cluster_centers_[kmeans.labels_]
            quant_k_means_model = deepcopy(self)
            quant_k_means_model.set_from_concatenated_UV_truncs(torch.tensor(quant_k_means_UV_truncs, device=self.device))
            quant_k_means_model.update_weights()
        quant_k_means_model.eval()
        return quant_k_means_model

    # TODO: Delete this as it's now taken care of in the parent MLP class
    # def quant_k_means_size_in_bits(self, codeword_length: Optional[int]=None) -> int:
    #     """"Returns the number of bits required to specify the quantized model, including the codebook"""
    #     if codeword_length == 32:
    #         num_codewords = 0  # no quantization
    #     else:
    #         num_codewords = 2 ** codeword_length
    #     # Only the UV_truncs are quantized, not the biases
    #     UV_truncs_sizes = codeword_length * self.num_UV_trunc_vals
    #     S_truncs_sizes = 32 * self.num_S_trunc_vals
    #     biases_sizes = 32 * self.num_biases
    #     codebook_size = 32 * num_codewords
    #     return UV_truncs_sizes + S_truncs_sizes + biases_sizes + codebook_size

    # TODO: Delte this as it's now taken care of in the parent MLP class
    # def get_KL_of_quant_k_means_model(self, codeword_length: int) -> torch.Tensor:
    #     """The KL between a point mass on the compressed representation of the model, with bit length
    #     b = self.quantized_size_in_bits(), against a uniform prior over the 2**b possible bit strings.
    #     This assumes that the ranks and codeword_length are fixed. A union bound should later account
    #     for any choice made over the ranks and codeword_length. Note there's no sigma as the posterior
    #     is a point mass on the final classifier."""
    #     return self.quant_k_means_size_in_bits(codeword_length=codeword_length) + torch.log(torch.tensor(2))

    def get_quant_trunc_model(self, exponent_bits: int, mantissa_bits: int) -> MLP:
        """Returns a new model with the weights truncated to the given number of bits."""
        quant_trunc_model = deepcopy(self)
        concatenated_UV_truncs = self.get_concatenated_UV_truncs()
        quant_trunc_UV_truncs = truncate(x=concatenated_UV_truncs, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)
        quant_trunc_model.set_from_concatenated_UV_truncs(quant_trunc_UV_truncs)
        quant_trunc_model.eval()
        return quant_trunc_model

    # TODO: Delete this as it's now taken care of in the parent MLP class
    # def quant_trunc_size_in_bits(self, exponent_bits: int, mantissa_bits: int) -> int:
    #     """Returns the number of bits required to specify the truncated model."""
    #     UV_truncs_sizes = (1 + exponent_bits + mantissa_bits) * self.num_UV_trunc_vals
    #     S_truncs_sizes = 32 * self.num_S_trunc_vals
    #     biases_sizes = 32 * self.num_biases
    #     return UV_truncs_sizes + S_truncs_sizes + biases_sizes

    # def get_KL_of_quant_trunc_model(self, exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    #     return self.quant_trunc_size_in_bits(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits) + torch.log(torch.tensor(2))
