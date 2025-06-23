import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)


class Model:
    def __init__(self, input_size):
        self.device = torch.device("cpu") # On rasp no GPU :/
        self.model = Classifier(input_size).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def get_weights(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_weights(self, weights):
        params = zip(self.model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, X, y, epochs=10, batch_size=8):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
       	    y_tensor = torch.tensor(y, dtype=torch.long)

       	    logits = self.model(X_tensor)
       	    loss = self.loss_fn(logits, y_tensor)

            preds = torch.argmax(logits, dim=1)
            correct = (preds == y_tensor).sum().item()
       	    total = y_tensor.size(0)
       	    accuracy = correct / total
        return loss.item(), accuracy
