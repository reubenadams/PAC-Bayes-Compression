import torch

from models import MLP, ParameterDataset


base_model_dims = [784, 128, 10]
base_model = MLP(base_model_dims)
base_model.load_state_dict(torch.load('mnist_mlp.t', weights_only=True))
print(f"Number of parameters in base model: {base_model.num_parameters()}")

hyper_model_dims = [3, 128, 1]
hyper_model = MLP(hyper_model_dims)
print(f"Number of parameters in hyper model: {hyper_model.num_parameters()}")


param_dataset = ParameterDataset(base_model)
param_dataset.return_transformed = True
param_dataloader = torch.utils.data.DataLoader(param_dataset, batch_size=128, shuffle=True)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(hyper_model.parameters(), lr=0.001)
num_epochs = 5

# all_xs = []
# for x, _ in param_dataloader:
#     all_xs.append(x)
# all_xs = torch.cat(all_xs, dim=0)
# print(all_xs.float().mean(dim=0))
# print(all_xs.float().std(dim=0))


# Training loop
total_loss = torch.tensor(0.)
for x, y in param_dataloader:
    outputs = hyper_model(x)
    total_loss += criterion(outputs, y)
print(f'Epoch [{0}/{num_epochs}], Total Loss: {total_loss.item():.4f}')

for epoch in range(num_epochs):
    for x, y in param_dataloader:
        outputs = hyper_model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss = torch.tensor(0.)
    for x, y in param_dataloader:
        outputs = hyper_model(x)
        total_loss += criterion(outputs, y)
    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.4f}')
