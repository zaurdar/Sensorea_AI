import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import cloudpickle

class preprocessing_pipeline:
    def __init__(self, tree=None):
        def fnct(x):
            return x
        if tree is None:
            tree = {"father":{"lil_bro":"dawn", "fnct":fnct,"param":{}},"dawn":{"lil_bro":"dawn", "fnct":fnct,"param":{}}}
        self.tree = tree
    def join(self,fnct,last_node,node,param):
        self.tree[last_node]["lil_bro"] = node
        self.tree[node] = {"lil_bro":"dawn","fnct":fnct,"param":param}
    def fit(self):
        temp ="father"
        pipe = lambda x: x
        while self.tree[temp]["lil_bro"] != "dawn":
            temp = self.tree[temp]["lil_bro"]
            f = self.tree[temp]["fnct"]
            param = self.tree[temp]["param"]
            pipe = lambda x, pipe=pipe, f=f, param=param: f(pipe(x), **param)
        return pipe
    def rendering(self,dummy):
        lines = []
        temp = "father"
        step = 0
        out_dummy = self.tree[temp]["fnct"](dummy)
        lines.append(f"└── [{step}──node : {temp}────────────────────────────────")
        while self.tree[temp]["lil_bro"] != "dawn":
            temp = self.tree[temp]["lil_bro"]
            step+=1
            dummy = out_dummy
            params = self.tree[temp]["param"]
            out_dummy = self.tree[temp]["fnct"](dummy, **params)
            lines.append(f"├── [{step}──node : {temp}──input_shape : [{dummy.shape}]──output_shape : [{out_dummy.shape}]")
        for line in lines:
            print(line)
    def save(self, path):
        with open(path, "wb") as f:
            cloudpickle.dump(self.tree, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tree = cloudpickle.load(f)
        return self
class pytorch_compiler:
    def __init__(self,model,optimizer,loss,epochs,batch_size):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
    def fit(self,inputs,targets):
        dataset = TensorDataset(inputs, targets)
        indices = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        model = self.model
        device = self.device
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        batch_size = self.batch_size

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        print(f"├── nombre de fichiers d'entrainement : {len(train_set)}")
        print(f"├── nombre de fichiers de test : {len(test_set)}")
        criterion = self.loss
        optimizer = self.optimizer
        train_losses = []
        val_losses = []
        model.to(self.device)
        pbar = tqdm(range(self.epochs), desc="training")
        for epoch in pbar:
            model.train()
            running_train_loss = 0.0
            n_seen = 0
            for X, Y in train_dataloader:
                X = X.to(device)
                Y = Y.to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                bs = X.size(0)
                running_train_loss += loss.item() * bs
                n_seen += bs

            epoch_train_loss = running_train_loss / max(1, n_seen)
            train_losses.append(epoch_train_loss)
            running_val_loss = 0.0
            n_seen = 0
            model.eval()
            with torch.no_grad():
                for X, Y in test_dataloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    output = model(X)
                    loss = criterion(output, Y)
                    bs = X.size(0)
                    running_val_loss += loss.item() * bs
                    n_seen += bs
            epoch_val_loss = running_val_loss / max(1, n_seen)
            val_losses.append(epoch_val_loss)
            pbar.set_postfix(train_loss=epoch_train_loss, val_loss=epoch_val_loss)
        return model, train_losses, val_losses