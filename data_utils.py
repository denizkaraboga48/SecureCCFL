import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train, test

def split_non_iid(dataset, num_clients=15, num_classes=10, classes_per_client=2,
                  min_per_class=200, batch_size=32, seed=42):
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    rng = np.random.default_rng(seed)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, y) in enumerate(dataset):
        class_indices[int(y)].append(idx)

    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    client_classes = [rng.choice(np.arange(num_classes), size=classes_per_client, replace=False)
                      for _ in range(num_clients)]

    client_data = [[] for _ in range(num_clients)]

    for i, cls_set in enumerate(client_classes):
        for cls in cls_set:
            take = min(min_per_class, len(class_indices[cls]))
            if take > 0:
                client_data[i].extend(class_indices[cls][:take])
                class_indices[cls] = class_indices[cls][take:]

    leftovers = []
    for cls in range(num_classes):
        leftovers.extend(class_indices[cls])
    rng.shuffle(leftovers)
    p = 0
    while leftovers:
        client_data[p % num_clients].append(leftovers.pop())
        p += 1

    for i in range(num_clients):
        if len(client_data[i]) < batch_size:
            pass

    loaders = []
    for idxs in client_data:
        subset = Subset(dataset, idxs)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return loaders
