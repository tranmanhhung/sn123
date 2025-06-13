FEATURE_LENGTH = 100

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.tanh(self.layer3(x))



def salience(history_dict, btc_prices, hidden_size=64, lr=1e-3):
    import torch
    NUM_UIDS = 256
    emb_dim = len(next(iter(history_dict.values()))[0])
    T = min(len(next(iter(history_dict.values()))), len(btc_prices))

    X = torch.zeros(T, NUM_UIDS * emb_dim, dtype=torch.float32)
    for uid in range(NUM_UIDS):
        h = torch.tensor(history_dict[uid][:T], dtype=torch.float32)
        X[:, uid * emb_dim : (uid + 1) * emb_dim] = h
    y = torch.tensor(btc_prices[:T], dtype=torch.float32).view(-1, 1)

    def run_model(mask_uid=None):
        model = MLP(X.shape[1], hidden_size, 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        total_loss = 0.0
        for i in range(T):
            inp = X[i : i + 1]
            if mask_uid is not None:
                s = mask_uid * emb_dim
                inp = inp.clone()
                inp[:, s : s + emb_dim] = 0.0
            pred = model(inp)
            loss = crit(pred, y[i : i + 1])
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return total_loss / T

    full_loss = run_model()
    losses = [run_model(uid) for uid in range(NUM_UIDS)]
    deltas = torch.tensor([l - full_loss for l in losses]).clamp(min=0.0)
    return (deltas / deltas.sum()).tolist() if deltas.sum() > 0 else [0.0]*NUM_UIDS
