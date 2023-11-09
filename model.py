import torch
from torch import nn

class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        
        super(LSTMClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.device = torch.device("cpu")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            # Set initial hidden and cell states 
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 

        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMPrediction, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        outputs = []
        for _ in range(96):  # predict next 96 values
            out, (h0, c0) = self.lstm(x, (h0.detach(), c0.detach()))  
            out = self.fc(out[:, -1, :])  # only take the last output
            outputs.append(out.unsqueeze(1))

            # Use current output to predict next value
            x = torch.cat((x[:, :, 1:], out.unsqueeze(1)), dim=2)

        return torch.cat(outputs, dim=1)  # shape: (batch_size, 96, 1)


class RNNPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNPrediction, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        outputs = []
        for _ in range(96):  # predict next 96 values
            out, h0 = self.rnn(x, h0.detach())  
            out = self.fc(out[:, -1, :])  # only take the last output
            outputs.append(out.unsqueeze(1))

            # Use current output to predict next value
            x = torch.cat((x[:, :, 1:], out.unsqueeze(1)), dim=2)

        return torch.cat(outputs, dim=1)  # shape: (batch_size, 96, 1)

