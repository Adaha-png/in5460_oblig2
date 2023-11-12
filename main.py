from sklearn.metrics import confusion_matrix
from model import RNNClassifier, LSTMClassifier, LSTMPrediction, RNNPrediction
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from dataset import CustomDataset

rnnLossList = []
lstmLossList = []
rnnPredList = []
lstmPredList = []
rnnGroundTruth = []
lstmGroundTruth = []
rnnAccList = []
lstmAccList = []

def one_hot_encode(x, num_classes):
    return np.eye(num_classes)[x]

def main():
# Let's define some hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device = torch.device("cpu")
    torch.set_default_device(device)
    input_size = 96
    hidden_size = 128 # hidden size
    num_layers = 2 # number of LSTM layers
    num_classes = 10 # number of outputs
    lstm = True
    classification = True
    if classification:
        if lstm:
            model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        else:
            model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
    else:
        if lstm:
            model = LSTMPrediction(input_size*7, hidden_size, num_layers)
        else:
            model = RNNPrediction(input_size*7, hidden_size, num_layers)
    run_centralised(epochs = 1, lr = 0.01, model = model, classification = classification, lstm = lstm)


def train(net, trainloader, optimizer, epochs, classification, lstm):
    """Train the network on the training set."""
    if classification:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    cm = np.array([[0,0],[0,0]])
    net.train()
    timestep=0
    correct=0
    for e in range(epochs):
        print(f"{e+1}/{epochs}", end = "\r")
        for consumption, labels in trainloader:
            timestep+=1
            optimizer.zero_grad()
            if classification:
                vals = net(consumption)
                _, predicted = torch.max(vals.data, 1)
                correct += (predicted == labels)
                loss = criterion(vals, labels)
                if lstm:
                    lstmAccList.append(correct/timestep)
                else:
                    rnnAccList.append(correct/timestep)
            else:
                vals = net(consumption)
                loss = criterion(vals, labels.float())
                if lstm:
                    lstmLossList.append(loss)
                    lstmPredList.append(vals)
                    lstmGroundTruth.append(labels)
                else:
                    rnnLossList.append(loss)
                    rnnPredList.append(vals)
                    rnnGroundTruth.append(labels)
            loss.backward()
            optimizer.step()
    if classification:
        with torch.no_grad():
            for consumption, labels in trainloader:
                preds = net(consumption)
                num_classes = 10
                one_hot_arr = one_hot_encode(labels, num_classes)
                p_new = np.zeros(10)
                p_new[preds.argmax()] = 1
                cm = cm + confusion_matrix(one_hot_arr, p_new)/10
            cm = cm/len(trainloader.dataset)
    return net, cm


def test(net, testloader, classification):
    """Validate the network on the entire test set."""
    if classification:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    correct, loss = np.zeros(10), 0.0
    net.eval()
    cm = np.array([[0,0],[0,0]])
    with torch.no_grad():
        for consumption, labels in testloader:
            outputs = net(consumption)
            if classification:
                num_classes = 10
                one_hot_arr = one_hot_encode(labels, num_classes)
                p_new = np.zeros(10)
                p_new[outputs.argmax()] = 1
                cm = cm + confusion_matrix(one_hot_arr, p_new)/10
                _, predicted = torch.max(outputs.data, 1)
                correct[labels] += (predicted == labels)
            loss += criterion(outputs, labels).item()
    if classification:
        cm = cm/len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)*10
    return loss, accuracy, cm


def collate_fn(batch): 
    # each item in batch will be a tuple (input, target)
    # the input is a multi-dimensional tensor
    data = torch.stack([item[0] for item in batch]).unsqueeze(0)
    target = torch.stack([item[1] for item in batch])
    return data, target


def run_centralised(epochs: int, lr: float, momentum: float = 0.9, model = None, classification = True, lstm = True):
    """A minimal (but complete) training loop"""
    

    if classification:
        # define optimiser with hyperparameters supplied
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        inds = random.sample(list(np.arange(int(35136/96))), k = round(35136//96 * 0.8))
    else:
        # define optimiser with hyperparameters supplied
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        inds = random.sample(list(np.arange(int(35136/96 - 7))), k = round((35136//96 - 7)* 0.8))

    # get dataset and construct a dataloaders
    trainset, testset = CustomDataset(True, inds, classification), CustomDataset(False, inds, classification)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=10, collate_fn = collate_fn)
    testloader = DataLoader(testset, batch_size=1, collate_fn = collate_fn)
    # train for the specified number of epochs
    trained_model, cm_train = train(model, trainloader, optim, epochs, classification, lstm)
    loss, accuracy, cm_test = test(trained_model, testloader, classification)
    if classification:
        print(f"{loss = }")
        print(f"{accuracy = }")
        print(f"{cm_train = }")
        print(f"{cm_test = }")
        if lstm:
            with open("lstmAcclist.txt", "w") as f:
                for i in lstmAccList:
                    f.write("%s; " % i.item())
            torch.save(trained_model.state_dict(),"classification/lstm.pt")
        else:
            with open("rnnAcclist.txt", "w") as f:
                for i in rnnAccList:
                    f.write("%s; " % i.item())
            torch.save(trained_model.state_dict(),"classification/rnn.pt")
    else:
        print(f"{loss = }")
        if lstm:
            with open("lstmLosslist.txt", "w") as f:
                for i in lstmLossList:
                    f.write("%s; " % i.item())
            with open("lstmPredlist.txt", "w") as f:
                for i in lstmPredList:
                    new = i.view(-1)
                    for j in new:
                        f.write("%s; " % j.item())
            with open("lstmGroundTruth.txt", "w") as f:
                for i in lstmGroundTruth:
                    new = i.view(-1)
                    for j in new:
                        f.write("%s; " % j.item())
            torch.save(trained_model.state_dict(),"prediction/lstm.pt")
        else:
            with open("rnnLosslist.txt", "w") as f:
                for i in rnnLossList:
                    f.write("%s; " % i.item())
            with open("rnnPredlist.txt", "w") as f:
                for i in rnnPredList:
                    new = i.view(-1)
                    for j in new:
                        f.write("%s; " % j.item())
            with open("rnnGroundTruth.txt", "w") as f:
                for i in rnnGroundTruth :
                    new = i.view(-1)
                    for j in new:
                        f.write("%s; " % j.item())
            torch.save(trained_model.state_dict(),"prediction/rnn.pt")
            



if __name__=="__main__":
    main()
