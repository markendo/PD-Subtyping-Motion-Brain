import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, classification_report, roc_auc_score

class mlp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 3)
        
    def forward(self, x):
        intermediate = nn.ReLU()(self.fc1(x))
        output = self.fc2(intermediate)
        return output, intermediate

def train_eval(X, y, use_random_connections=False, num_epochs=100, num_runs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f1_scores = np.zeros(num_runs)
    precision_scores = np.zeros(num_runs)
    recall_scores = np.zeros(num_runs)
    auc_scores = np.zeros(num_runs)

    for permutation in range(num_runs):
        leave_one_out = LeaveOneOut()

        leave_one_out_preds = []
        leave_one_out_probs = []

        if use_random_connections:
            X_run = X[:, np.random.choice(np.shape(X)[1], size=100, replace=False)]
        else:
            X_run = X

        for train_index, test_index in leave_one_out.split(X_run):
            train_X = torch.tensor(X_run[train_index,:], dtype=torch.float).to(device)
            train_y = (torch.tensor(y[train_index], dtype=torch.long)).to(device)
            test_X = torch.tensor(X_run[test_index,:], dtype=torch.float).to(device)
            test_y = (torch.tensor(y[test_index], dtype=torch.long)).to(device)

            # model
            num_network_connections = np.shape(train_X)[1]
            model = mlp(num_network_connections)
            model.to(device)
            model.train()

            # optimizer
            optimizer = Adam(model.parameters())

            # weighted loss function
            weights = torch.bincount(train_y.int()).float()
            weights = weights / weights.sum() # turn into percentage
            weights = 1.0 / weights # inverse
            weights = weights / weights.sum()
            loss_weights = weights
            loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

            for epoch in range(1, num_epochs+1):
                optimizer.zero_grad()
                shuffled_indices = torch.randperm(train_y.size()[0])
                train_X_shuffle = train_X[shuffled_indices]
                train_y_shuffle = train_y[shuffled_indices]

                pred, inter = model(train_X_shuffle)
                loss=loss_fn(pred,train_y_shuffle)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred, inter = model(test_X)
                inter = inter.squeeze().cpu().numpy()

                leave_one_out_preds.append(torch.argmax(pred, dim=1)[0].item())
                leave_one_out_probs.append(torch.nn.Softmax(dim=0)(pred[0]).cpu().numpy())

        perm_f1_score = f1_score(y, leave_one_out_preds, average='macro')
        f1_scores[permutation] = perm_f1_score

        leave_one_out_probs = np.asarray(leave_one_out_probs)
        perm_auc_score = roc_auc_score(y, leave_one_out_probs, multi_class='ovr')
        auc_scores[permutation] = perm_auc_score

        report = classification_report(y, leave_one_out_preds, output_dict=True)

        precision_scores[permutation] = report['macro avg']['precision']
        recall_scores[permutation] = report['macro avg']['recall']

    return auc_scores, f1_scores, precision_scores, recall_scores