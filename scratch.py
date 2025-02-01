import torch
import torch.nn.functional as F


loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="none")


pred = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
true = torch.tensor([[0.3, 0.6], [0.8, 0.2]])


pred_log = torch.log(pred)
true_log = torch.log(true)

loss = loss_fn(pred_log, true_log)
print(loss, loss.sum(-1).mean())
print()

kl_true_pred_all = true * torch.log(true / pred)
kl_true_pred = kl_true_pred_all.sum(-1).mean()
print(kl_true_pred_all, kl_true_pred)
print()


loss_fn = torch.nn.MSELoss(reduction="none")
loss = loss_fn(pred, true)
print(loss, loss.sum(-1).mean())
print()

loss = (pred - true).pow(2).sum(-1).mean()
print(loss)

loss = (pred - true).pow(2).sum(0).mean()
print(loss)