# data_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train, test

def split_non_iid(dataset, num_clients=15, num_classes=10, classes_per_client=2):
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_data = {i: [] for i in range(num_clients)}
    all_classes = np.arange(num_classes)
    np.random.seed(42)

    for client in range(num_clients):
        selected = np.random.choice(all_classes, classes_per_client, replace=False)
        for cls in selected:
            selected_idx = np.random.choice(class_indices[cls], size=len(class_indices[cls]) // num_clients, replace=False)
            client_data[client].extend(selected_idx)
            class_indices[cls] = list(set(class_indices[cls]) - set(selected_idx))

    client_loaders = []
    for indices in client_data.values():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_loaders.append(loader)

    return client_loaders
