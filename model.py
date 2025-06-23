# MIT License
#
# Copyright (c) 2024 MANTIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import config

FEATURE_LENGTH = config.FEATURE_LENGTH

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

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



def salience(history_dict, btc_prices, hidden_size=config.HIDDEN_SIZE, lr=config.LEARNING_RATE, loss_type: str = "mae"):
    """Compute salience scores for each UID.

    history_dict: Dict[int, List[List[float]]]
        Maps UID -> sequence of embedding vectors.
    btc_prices: List[float]
        Target BTC percentage-change series.
    hidden_size: int
        Hidden layer width for the small MLP used as a proxy model.
    lr: float
        Learning rate for the proxy model optimiser.
    loss_type: str
        Loss type for the proxy model. Options: "mae" (default), "mape", "mse".
    """
    import torch
    NUM_UIDS = config.NUM_UIDS
    emb_dim = len(next(iter(history_dict.values()))[0])
    T = min(len(next(iter(history_dict.values()))), len(btc_prices))

    logger.info("Starting salience computation")
    logger.debug(f"NUM_UIDS={NUM_UIDS}, emb_dim={emb_dim}, T={T}")

    X = torch.zeros(T, NUM_UIDS * emb_dim, dtype=torch.float32)
    for uid in range(NUM_UIDS):
        h = torch.tensor(history_dict[uid][:T], dtype=torch.float32)
        X[:, uid * emb_dim : (uid + 1) * emb_dim] = h
    y = torch.tensor(btc_prices[:T], dtype=torch.float32).view(-1, 1)

    logger.debug(f"Feature matrix shape: {X.shape}, target vector shape: {y.shape}")

    def run_model(mask_uid=None):
        """Return average prediction loss using horizon-delayed updates.

        If `mask_uid` is not None the corresponding embedding slice is
        zeroed out at inference *and* training time to estimate its
        marginal contribution (salience).
        """

        model = MLP(X.shape[1], hidden_size, 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.L1Loss()

        lag = config.LAG

        buf_x: list[torch.Tensor] = []
        buf_y: list[torch.Tensor] = []

        total_loss = 0.0

        for t in range(T):

            inp = X[t : t + 1]
            if mask_uid is not None:
                s = mask_uid * emb_dim
                inp = inp.clone()
                inp[:, s : s + emb_dim] = 0.0

            pred = model(inp)
            loss_eval = crit(pred, y[t : t + 1])
            total_loss += loss_eval.item()


            buf_x.append(inp.detach())
            buf_y.append(y[t : t + 1].detach())


            if len(buf_x) > lag:
                xb = buf_x.pop(0)
                yb = buf_y.pop(0)

                opt.zero_grad()
                train_loss = crit(model(xb), yb)
                train_loss.backward()
                opt.step()

        return total_loss / T

    full_loss = run_model()
    logger.debug(f"Full (no-mask) model loss: {full_loss:.6f}")

    losses = []
    for uid in range(NUM_UIDS):
        if all(all(v == 0 for v in vec) for vec in history_dict[uid]):
            losses.append(full_loss) 
            continue
        l = run_model(uid)
        losses.append(l)
        if uid % 32 == 0:
            logger.debug(f"Computed masked loss for UID {uid}: {l:.6f}")

    deltas = torch.tensor([l - full_loss for l in losses]).clamp(min=0.0)
    logger.debug(f"Delta losses tensor: {deltas}")

    weights = (deltas / deltas.sum()).tolist() if deltas.sum() > 0 else [0.0] * NUM_UIDS
    logger.info("Salience computation complete")

    return weights
