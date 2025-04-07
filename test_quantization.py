from models import MLP
from sklearn.cluster import KMeans
import numpy as np
import torch
from copy import deepcopy

# X = np.arange(12).reshape(-1, 1)
# print(X)

model1 = MLP(dimensions=[1, 2, 1], activation="relu")
model2 = deepcopy(model1)
model1_weights = model1.get_concatenated_weights()
model2_weights = model2.get_concatenated_weights()
print("model1 weights")
print(model1_weights)

print("model2 weights")
print(model2_weights)

kmeans = KMeans(n_clusters=2, random_state=0).fit(model1_weights)
quantized_weights = kmeans.cluster_centers_[kmeans.labels_]
model1.set_from_concatenated_weights(torch.tensor(quantized_weights))
print("model1 quantized weights")
print(model1.get_concatenated_weights())

model3 = model1.get_quantized_model(codeword_length=1)
print("model3 quantized weights")
print(model3.get_concatenated_weights())
