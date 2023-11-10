from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, train, inds, classification):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.device = torch.device("cpu")
        self.cols = ["AC", "Dish washer", "Washing Machine", 
                     "Dryer", "Water heater", "TV", 
                     "Microwave", "Kettle", "Lighting", "Refrigerator"]
        self.inds = inds
        self.train = train
        self.households = 1
        self.classific = classification
        self.X, self.y = self.load_data()

    def load_data(self):
        prediction = []
        classification = []
        for i in range(self.households):
            df = pd.read_excel("dataset.xlsx",sheet_name=i)
            prediction.append(df[self.cols].sum(axis='columns').to_numpy())
            classification.append(df[self.cols].to_numpy())
            
        classification = np.array(classification)
        prediction = np.array(prediction)
        if self.train:
            if self.classific:
                classification_x = [classification[k,i*96:96*(i+1),j] for k in range(self.households) for j in range(len(self.cols)) for i in self.inds]
                classification_y = [j//(len(classification_x)//10) for j in range(len(classification_x))]
            else:
                prediction_x = [prediction[k,i*96:96*(i+7)] for k in range(self.households) for i in self.inds]
                prediction_y = [prediction[k,(i+7)*96:96*(i+8)] for k in range(self.households) for i in self.inds]
        else:
            if self.classific:
                classification_x = [classification[k,i*96:96*(i+1),j] for j in range(len(self.cols)) for i in np.arange(35136//96) if i not in self.inds for k in range(self.households)]
                classification_y = [j//(len(classification_x)//10) for j in range(len(classification_x))]
            else:
                prediction_x = [prediction[k,i*96:96*(i+7)] for k in range(self.households) for i in np.arange(35136//96 - 7) if  i not in self.inds]
                prediction_y = [prediction[k,(i+7)*96:96*(i+8)] for k in range(self.households) for i in np.arange(35136//96 - 7) if i not in self.inds]

        if self.classific:
            return torch.Tensor(np.array(classification_x)).to(self.device), torch.Tensor(classification_y).to(device = self.device, dtype = torch.long)
        else:
            prediction_x, prediction_y = torch.Tensor(np.array(prediction_x)).to(self.device), torch.Tensor(np.array(prediction_y)).to(device = self.device, dtype = torch.long)
            p_min, p_max = prediction_x.min(), prediction_x.max()
            new_min, new_max = 0, 1
            prediction_y = (prediction_y-p_min)/(p_max-p_min)*(new_max-new_min) + new_min
            prediction_x = (prediction_x-p_min)/(p_max-p_min)*(new_max-new_min) + new_min
            return prediction_x, prediction_y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]).to(device = self.device), torch.Tensor(self.y[idx]).to(device = self.device)
