from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, train, inds):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.device = torch.device("cpu")
        self.cols = ["AC", "Dish washer", "Washing Machine", 
                     "Dryer", "Water heater", "TV", 
                     "Microwave", "Kettle", "Lighting", "Refrigerator"]
        self.inds = inds
        self.train = train
        self.X, self.y = self.load_data()


    def load_data(self):
        
        prediction = []
        classification = []
        for i in range(5):
            df = pd.read_excel("dataset.xlsx",sheet_name=i)
            prediction.append(df[self.cols].sum(axis='columns').to_numpy())
            classification.append(df[self.cols].to_numpy())
            
        classification = np.array(classification)
        if self.train:
            classification_x = [classification[k,i*96:96*(i+1),j] for j in range(len(self.cols)) for i in self.inds for k in range(5)]
            classification_y = [j%10 for j in range(len(classification_x))]
            #prediction = np.array([prediction[:,i*96:96*(i+1)] for i in self.inds])
        else:
            classification_x = [classification[k][i*96:96*(i+1)][j] for j in range(len(self.cols)) for i in np.arange(35136//96) if i not in self.inds for k in range(5)]
            classification_y = [int(j%10) for j in range(len(classification_x))]
            #prediction = np.array([prediction[:,i*96:(i+1)*96] for i in np.arange(35136//96) if i not in self.inds])

        return torch.Tensor(np.array(classification_x)).to(self.device), torch.Tensor(classification_y).to(device = self.device, dtype = torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]).to(device = self.device), torch.Tensor(self.y[idx]).to(device = self.device)
