from model import RNNClassifier, LSTMClassifier
import argparse
import pandas as pd
import numpy as np
# we naturally first need to import torch and torchvision
import torch
from torch.utils.data import DataLoader
import random
from dataset import CustomDataset

def main():
# Let's define some hyperparameters
    input_size = 96 # Input size should be based on your data
    hidden_size = 128 # hidden size
    num_layers = 2 # number of LSTM layers
    num_classes = 10 # number of outputs

    lstm_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    rnn_model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
    
    run_centralised(epochs = 5, lr = 0.001, model = lstm_model)

def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for consumption, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(consumption), labels)
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def run_centralised(epochs: int, lr: float, momentum: float = 0.9, model = None):
    """A minimal (but complete) training loop"""
    def collate_fn(batch):
        
        # each item in batch will be a tuple (input, target)
        # the input could be a multi-dimensional tensor
        data = torch.stack([item[0] for item in batch])

        # the target could be a single value, so we just construct a tensor out of them
        target = torch.stack([item[1] for item in batch])

        return data, target

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    inds = random.sample(list(np.arange(int(35136/96))), k = round(35136//96 * 0.8))
    

    # get dataset and construct a dataloaders
    trainset, testset = CustomDataset(True, inds), CustomDataset(False, inds)  # Assuming same dataset can be used for training and testing.
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, collate_fn = collate_fn)
    testloader = DataLoader(testset, batch_size=128)
    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")


if __name__=="__main__":
    main()
