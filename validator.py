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

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import queue
import asyncio

import bittensor as bt
import torch
import requests
from dotenv import load_dotenv

import config
from cycle import get_miner_payloads
from model import salience as sal_fn
from storage import DataLog

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.expanduser("~/new_system_mantis")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "main.log"), mode="a"),
    ],
)

weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "weights.log"), mode="a")
)

# Silence noisy loggers
for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
DATALOG_PATH = os.path.expanduser("~/mantis_datalog.pkl.gz")
# How often to process pending (undecrypted) payloads.
PROCESS_INTERVAL = 10  # blocks
# How often to save the entire datalog to disk.
SAVE_INTERVAL = 100  # blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    args = p.parse_args()

    # --- Bittensor objects ----
    sub = bt.subtensor(network=args.network)
    wallet = bt.wallet(name=getattr(args, "wallet.name"), hotkey=getattr(args, "wallet.hotkey"))
    mg = bt.metagraph(netuid=args.netuid, network=args.network, sync=True)

    # --- Load the DataLog ---
    datalog = DataLog.load(DATALOG_PATH)

    last_block = sub.get_current_block()
    next_process = last_block + PROCESS_INTERVAL
    next_save = last_block + SAVE_INTERVAL
    next_task = last_block + config.TASK_INTERVAL
    weight_thread: threading.Thread | None = None

    while True:
        try:
            current_block = sub.get_current_block()
            if current_block == last_block:
                time.sleep(1)
                continue

            last_block = current_block

            # Only sample and process every SAMPLE_STEP blocks.
            if current_block % config.SAMPLE_STEP != 0:
                continue

            logging.info(f"🪙 Sampled block {current_block}")

            # --- Sync metagraph ---
            if current_block % 100 == 0:
                mg.sync(subtensor=sub)
                logging.info("Metagraph synced.")

            # --- Get data and append to log ---
            try:
                price = float(
                    requests.get(
                        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                        timeout=5,
                    ).json()["price"]
                )
            except Exception as e:
                logging.error(f"Failed to fetch BTC price: {e}")
                continue  # Skip this block if we can't get a price

            payloads = get_miner_payloads(netuid=args.netuid, mg=mg)
            datalog.append_step(current_block, price, payloads)

            # --- Process pending payloads ---
            if current_block >= next_process and len(datalog.blocks) >= 70:
                logging.info("Decrypting and processing pending payloads...")
                asyncio.run(datalog.process_pending_payloads())
                next_process = current_block + PROCESS_INTERVAL

            # --- Set weights in a background thread ---
            if current_block >= next_task and (
                weight_thread is None or not weight_thread.is_alive()
            ):

                def worker(block_snapshot: int, metagraph: bt.metagraph):
                    weights_logger.info(
                        f"=== Weight computation start | block {block_snapshot} ==="
                    )
                    training_data = datalog.get_training_data()
                    if not training_data:
                        weights_logger.warning("Not enough data to compute salience.")
                        return

                    history, btc_returns = training_data
                    if not history:
                        weights_logger.warning("Training data was empty.")
                        return

                    sal = sal_fn(history, btc_returns)
                    if not sal:
                        weights_logger.info("Salience unavailable.")
                        return

                    # Set weights only for UIDs that are currently active in the metagraph.
                    uids_to_set = metagraph.uids.tolist()
                    w = torch.tensor(
                        [sal.get(uid, 0.0) for uid in uids_to_set],
                        dtype=torch.float32,
                    )
                    if w.sum() <= 0:
                        weights_logger.warning("Zero-sum weights, skipping.")
                        return

                    w_norm = w / w.sum()
                    sub.set_weights(
                        netuid=args.netuid,
                        wallet=wallet,
                        uids=uids_to_set,
                        weights=w_norm,
                        wait_for_inclusion=False,
                    )
                    weights_logger.info(
                        f"✅ Weights set at block {block_snapshot} (max={w_norm.max():.4f})"
                    )

                weight_thread = threading.Thread(
                    target=worker, args=(current_block, mg), daemon=True
                )
                weight_thread.start()
                next_task = current_block + config.TASK_INTERVAL

            # --- Periodically save the log ---
            if current_block >= next_save:
                datalog.save(DATALOG_PATH)
                next_save = current_block + SAVE_INTERVAL

        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
