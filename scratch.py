import torch
import torch.nn.functional as F


loss_fn = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")


pred = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
true = torch.tensor([[0.2, 0.8], [0.3, 0.7]])


pred_log = torch.log(pred)
true_log = torch.log(true)

loss = loss_fn(pred_log, true_log)
print(loss, loss.mean())
print()

kl_true_pred_all = true * torch.log(true / pred)
kl_true_pred = kl_true_pred_all.sum(-1).mean()
print(kl_true_pred_all, kl_true_pred)
print()

kl_pred_true_all = pred * torch.log(pred / true)
kl_pred_true = kl_pred_true_all.sum(-1).mean()
print(kl_pred_true_all, kl_pred_true)
print()
