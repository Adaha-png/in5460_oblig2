#from sklearn.metrics import confusion_matrix
from model import RNNClassifier, LSTMClassifier, LSTMPrediction, RNNPrediction
import argparse
import pandas as pd
import numpy as np
# we naturally first need to import torch and torchvision
import torch
from torch.utils.data import DataLoader
import random
from dataset import CustomDataset

rnnLossList = []
lstmLossList = []
rnnPredList = []
lstmPredList = []
rnnGrounTruth = []
lstmGroundTruth = []
rnnAccList = []
lstmAccList = []

def main():
# Let's define some hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device = torch.device("cpu")
    torch.set_default_device(device)
    input_size = 96
    hidden_size = 128 # hidden size
    num_layers = 2 # number of LSTM layers
    num_classes = 10 # number of outputs

    classification = True
    if classification:
        lstm_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        rnn_model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
    else:
        lstm_model = LSTMPrediction(input_size*7, hidden_size, num_layers)
        rnn_model = RNNPrediction(input_size*7, hidden_size, num_layers)

    run_centralised(epochs = 10, lr = 0.01, model = rnn_model, classification = classification)
    run_centralised(epochs = 10, lr = 0.01, model = lstm_model, classification = classification)

def train(net, trainloader, optimizer, epochs, classification):
    """Train the network on the training set."""
    if classification:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    #cm = np.array([[0,0],[0,0]])
    net.train()
    for _ in range(epochs):
        print(_)
        for consumption, labels in trainloader:
            optimizer.zero_grad()
            if classification:
                loss = criterion(net(consumption), labels)
            else:
                loss = criterion(net(consumption), labels.float())
            loss.backward()
            optimizer.step()
    if classification:
        with torch.no_grad():
            for consumption, labels in trainloader:
                preds = net(consumption)
                #cm += confusion_matrix(labels, preds)
            #cm /= len(trainloader.dataset)
    return net#, cm


def test(net, testloader, classification):
    """Validate the network on the entire test set."""
    if classification:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    correct, loss = 0, 0.0
    net.eval()
    #cm = np.array([[0,0],[0,0]])
    with torch.no_grad():
        for consumption, labels in testloader:
            outputs = net(consumption)
            if classification:
                ...
                #cm += confusion_matrix(labels, outputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels)
    if classification:
        ...
        #cm /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy#, cm


def run_centralised(epochs: int, lr: float, momentum: float = 0.9, model = None, classification = True):
    """A minimal (but complete) training loop"""
    def collate_fn(batch):
        
        # each item in batch will be a tuple (input, target)
        # the input is a multi-dimensional tensor
        data = torch.stack([item[0] for item in batch]).unsqueeze(0)

        target = torch.stack([item[1] for item in batch])

        return data, target

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if classification:
        inds = random.sample(list(np.arange(int(35136/96))), k = round(35136//96 * 0.8))
    else:
        inds = random.sample(list(np.arange(int(35136/96 - 7))), k = round((35136//96 - 7)* 0.8))

    # get dataset and construct a dataloaders
    trainset, testset = CustomDataset(True, inds, classification), CustomDataset(False, inds, classification)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=15, collate_fn = collate_fn)
    testloader = DataLoader(testset, batch_size=1, collate_fn = collate_fn)
    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, epochs, classification)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, testloader, classification)
    print(f"{loss = }")
    print(f"{accuracy = }")


if __name__=="__main__":
    main()
