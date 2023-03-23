import numpy as np

import torch 
from torch import nn

def disable_dropout(model):
    """ Function to disable the dropout layers during training """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

def enable_dropout(model):
    """ Function to enable the dropout layers during evaluation """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class MVENetworkWithDropout(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(MVENetworkWithDropout, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size), # 1st layer
            nn.ReLU(), # activation function
            nn.Dropout(p=dropout), # dropout layer
            nn.Linear(hidden_size, hidden_size), # 2nd layer
            nn.ReLU(), # activation function
            nn.Dropout(p=dropout), # dropout layer
            nn.Linear(hidden_size, 2), # output layer
        )

    def forward(self, x):
        preds = self.linear_relu_stack(x)
        mean = preds[:, 0]
        var = torch.abs(preds[:, 1]) # we use the absolute value of the variance to ensure it is positive
        return mean, var
    
def train(dataloader, model, epochs):

    """ 
    Function to train the model for a given number of epochs. 
    Note that the dropout layers are disabled during training.
    
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader object.
    model : torch.nn.Module
        Model to be trained.
    epochs : int
        Number of epochs to train the model for.
    """

    for epoch in range(epochs):
        device = torch.device("cpu")
        loss_fn = nn.GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        disable_dropout(model)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Make predictions
            mean, var = model(X)

            # Compute loss
            loss = loss_fn(y, mean, var)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} - loss: {loss.item():>7f}")


def predict(dataloader, model, n_samples=100):

    """
    Make predictions with total uncertainty estimated by combining aleatoric uncertainty (from MVE) 
    and epistemic uncertainty (from dropout).
    """

    device = torch.device("cpu")
    model.eval()
    enable_dropout(model)
    
    preds = np.empty((len(dataloader.dataset), n_samples))
    varsA = np.empty((len(dataloader.dataset), n_samples))
    
    for sample in range(n_samples):
        preds_sample, varsA_sample = [], []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred, varA = model(X)
            preds_sample.append(pred)
            varsA_sample.append(varA)


        preds_sample = torch.cat(preds_sample)
        varsA_sample = torch.cat(varsA_sample)

        preds[:, sample] = preds_sample.detach().numpy().squeeze()
        varsA[:, sample] = varsA_sample.detach().numpy().squeeze()

    means = torch.Tensor(preds.mean(axis=1))
    varsE = torch.Tensor(preds.var(axis=1))    
    vars = varsE + torch.Tensor(varsA.mean(axis=1))

    return means, vars