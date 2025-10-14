import torch
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
train_nodes, eval_nodes = get_graph_node_names(model)
print("Train nodes:")
print(train_nodes)
print("\nEval nodes:")
print(eval_nodes)
