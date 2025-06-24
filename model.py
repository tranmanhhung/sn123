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
import os, time

FEATURE_LENGTH = config.FEATURE_LENGTH

import torch
import torch.nn as nn
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
logger.info("Salience computations will run on %s", DEVICE)

try:
    _NUM_CPU = max(1, os.cpu_count() or 1)
    torch.set_num_threads(_NUM_CPU)
    torch.set_num_interop_threads(_NUM_CPU)
    logger.info("Torch thread pools set to %d", _NUM_CPU)
except Exception as e:
    logger.warning("Could not set torch thread counts: %s", e)

COMPILE_AVAILABLE = hasattr(torch, "compile")

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

if COMPILE_AVAILABLE:
    try:
        logger.info("Enabling torch.compile() for MLP")
        MLP = torch.compile(MLP)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("torch.compile unavailable or failed post-definition: %s", e)

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
    NUM_UIDS = config.NUM_UIDS


    active_uids = [uid for uid, h in history_dict.items() if isinstance(h, list) and len(h) > 0]
    emb_dim = config.FEATURE_LENGTH

    if not active_uids:
        logger.info("No active histories provided – returning all-zero salience weights")
        return [0.0] * NUM_UIDS

    T = min(min(len(history_dict[uid]) for uid in active_uids), len(btc_prices))

    logger.info("Starting salience computation")
    t0 = time.time()
    logger.debug(f"NUM_UIDS={NUM_UIDS}, emb_dim={emb_dim}, T={T}")

    X = torch.zeros(T, NUM_UIDS * emb_dim, dtype=torch.float32, device=DEVICE)
    for uid in range(NUM_UIDS):
        h_raw = history_dict.get(uid)
        if not isinstance(h_raw, list) or len(h_raw) == 0:
            continue
        h_tensor = torch.tensor(h_raw[:T], dtype=torch.float32, device=DEVICE)
        X[:, uid * emb_dim : (uid + 1) * emb_dim] = h_tensor
    y = torch.tensor(btc_prices[:T], dtype=torch.float32, device=DEVICE).view(-1, 1)

    logger.debug(f"Feature matrix shape: {X.shape}, target vector shape: {y.shape}")

    def run_model(mask_uid=None):
        """Return average prediction loss using horizon-delayed updates.

        Adds verbose timing and progress logging.
        """
        t_start = time.time()
        model = MLP(X.shape[1], hidden_size, 1).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.L1Loss()



        buf_x: list[torch.Tensor] = []
        buf_y: list[torch.Tensor] = []

        total_loss = 0.0

        lag = config.LAG  # number of *samples* separating training and inference

        for t in range(T):
            inp = X[t : t + 1]
            if mask_uid is not None:
                s = mask_uid * emb_dim
                inp = inp.clone()
                inp[:, s : s + emb_dim] = 0.0

            pred = model(inp)
            loss_eval = crit(pred, y[t : t + 1])
            total_loss += loss_eval.item()

            logger.debug(
                "step=%d | mask_uid=%s | pred=%.6f | target=%.6f | loss=%.6f",
                t,
                mask_uid,
                pred.item() if pred.numel() == 1 else float(pred.squeeze()[0].item()),
                y[t].item(),
                loss_eval.item(),
            )

            buf_x.append(inp.detach())
            buf_y.append(y[t : t + 1].detach())

            # ----------------------
            # Training (t - lag)
            # ----------------------
            if len(buf_x) > lag:
                xb = buf_x.pop(0)
                yb = buf_y.pop(0)

                opt.zero_grad()
                train_loss = crit(model(xb), yb)
                train_loss.backward()
                opt.step()

                logger.debug(
                    "backprop | mask_uid=%s | train_loss=%.6f | buffer_size=%d",
                    mask_uid,
                    train_loss.item(),
                    len(buf_x),
                )

        runtime = time.time() - t_start
        logger.debug("run_model(mask_uid=%s) finished in %.2fs | avg_loss=%.6f", mask_uid, runtime, total_loss / T)
        return total_loss / T

    full_loss = run_model()
    logger.debug(f"Full (no-mask) model loss: {full_loss:.6f}")

    losses = []
    for uid in range(NUM_UIDS):
        if uid % 32 == 0:
            logger.info("Computing masked loss for UID %d/%d", uid, NUM_UIDS)
        h_raw = history_dict.get(uid)
        if not isinstance(h_raw, list) or len(h_raw) == 0 or all(all(v == 0 for v in vec) for vec in h_raw):
            losses.append(full_loss) 
            continue
        l = run_model(uid)
        losses.append(l)
        if uid % 32 == 0:
            logger.debug("Masked loss for UID %d: %.6f", uid, l)

    deltas = torch.tensor([l - full_loss for l in losses]).clamp(min=0.0)
    logger.debug(f"Delta losses tensor: {deltas}")

    weights = (deltas.cpu() / deltas.sum().cpu()).tolist() if deltas.sum() > 0 else [0.0] * NUM_UIDS
    logger.info("Salience computation complete in %.2fs", time.time() - t0)

    return weights
