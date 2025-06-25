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
import ast
import json
import logging
import os
import threading
import time
from typing import Any
import queue

import bittensor as bt
import torch
from dotenv import load_dotenv

import config
from cycle import cycle, miner_data
from model import salience as sal_fn
import plaintext_utils as pt
from plaintext_utils import plaintext_miner_data

LOG_DIR = os.path.expanduser("~/mantis_logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "orchestrator.log"), mode="a"),
    ],
)

ws_handler = logging.FileHandler(os.path.join(LOG_DIR, "weights_salience.log"), mode="a")
ws_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(ws_handler)

logging.getLogger("model").addHandler(ws_handler)
logging.getLogger("model").setLevel(logging.DEBUG)

for noisy in ("websockets", "websockets.client", "websockets.server", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

def _init_from_archive() -> None:
    """Populate `miner_data` from a previous `archive.data(.gz)` snapshot."""
    import gzip, pathlib, requests

    paths = [
        pathlib.Path.home() / "archive.data.gz",
        pathlib.Path.home() / "archive.data",
    ]

    log = logging.getLogger("archive")
    data_bytes: bytes | None = None

    for p in paths:
        if p.exists():
            try:
                log.info("🗄️  Loading miner_data from %s", p)
                data_bytes = p.read_bytes()
                if p.suffix == ".gz":
                    data_bytes = gzip.decompress(data_bytes)
                break
            except Exception as e:
                log.warning("Failed to load %s: %s", p, e)

    if data_bytes is None:
        url = getattr(config, "ARCHIVE_URL", None)
        if url:
            try:
                log.info("🌐 Fetching miner_data archive from %s", url)
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data_bytes = r.content
            except Exception as e:
                log.warning("Remote archive fetch failed: %s", e)

    if data_bytes is None:
        log.info("No archive found – starting with empty miner_data")
        return

    try:
        try:
            import orjson  # type: ignore

            obj = orjson.loads(data_bytes)
        except ImportError:
            obj = json.loads(data_bytes.decode())

        miner_data.clear()
        for k, v in obj.items():
            try:
                miner_data[int(k)] = v
            except Exception:
                continue

        log.info("✓ Restored miner_data for %d UIDs from archive", len(miner_data))
        if miner_data:
            lengths = [
                len(rec.get("history"))
                for rec in miner_data.values()
                if isinstance(rec.get("history"), list)
            ]
            if lengths:
                log.info("Archive timestep length: %d", min(lengths))
    except Exception as e:
        log.warning("Failed to parse archive: %s", e)

    # PLAINTEXT_PATH = os.path.expanduser("~/plaintext.data.gz")
    # if not os.path.exists(PLAINTEXT_PATH):
    #     remote_pt = getattr(config, "ARCHIVE_URL_PLAINTEXT", None)
    #     if remote_pt:
    #         try:
    #             import requests
    #             logging.info("🌐 Fetching plaintext archive from %s", remote_pt)
    #             r = requests.get(remote_pt, timeout=30)
    #             r.raise_for_status()
    #             with open(PLAINTEXT_PATH, "wb") as f:
    #                 f.write(r.content)
    #             logging.info("✓ Saved remote plaintext archive to %s", PLAINTEXT_PATH)
    #         except Exception as e:
    #             logging.warning("Remote plaintext fetch failed: %s", e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=123)
    args = p.parse_args()

    _init_from_archive()

    pt.load_plaintext()
    
    logging.info("Starting initial plaintext update in the background.")
    initial_plaintext_thread = threading.Thread(target=pt.update_plaintext, args=(miner_data,), daemon=True)
    initial_plaintext_thread.start()

    sub = bt.subtensor(network=args.network)
    wallet = bt.wallet(name=getattr(args, "wallet.name"), hotkey=getattr(args, "wallet.hotkey"))

    netuid = args.netuid

    weight_queue: "queue.Queue[tuple[int, list[int], list[float]]]" = queue.Queue()

    # try:
    #     mg_start = bt.metagraph(netuid=netuid, network=args.network, lite=True, sync=True)
    #     min_hist = min(
    #         (
    #             len(rec.get("history"))
    #             for rec in miner_data.values()
    #             if isinstance(rec.get("history"), list)
    #         ),
    #         default=0,
    #     )

    #     if min_hist >= 2 * config.LAG + 1:
    #         initial_sal = pt.compute_salience_from_plaintext(sal_fn)
    #     else:
    #         initial_sal = None
    #     if initial_sal:
    #         w_init = torch.tensor([initial_sal[uid] if uid < len(initial_sal) else 0.0 for uid in mg_start.uids])
    #         if w_init.sum() > 0:
    #             w_init = w_init / w_init.sum()
    #             sub.set_weights(
    #                 netuid=netuid,
    #                 wallet=wallet,
    #                 uids=mg_start.uids,
    #                 weights=w_init,
    #                 wait_for_inclusion=False,
    #             )
    #             weights_logger.info("✓ Initial set_weights submitted immediately after archive load (max=%.4f)", w_init.max())
    #     else:
    #         weights_logger.info("Archive loaded but not enough data yet for initial salience computation")
    # except Exception as e:
    #     weights_logger.warning("Initial salience/weights failed: %s", e)

    last_block = sub.get_current_block()
    next_task = last_block + config.TASK_INTERVAL
    weight_thread: threading.Thread | None = None
    next_plaintext_update = last_block + config.PLAINTEXT_UPDATE_INTERVAL
    plaintext_update_task: threading.Thread | None = None

    while True:
        blk = sub.get_current_block()
        if blk != last_block:
            if blk % config.SAMPLE_STEP != 0:
                last_block = blk
                time.sleep(1)
                continue

            logging.info("🪙 New block %s", blk)

            mg: bt.metagraph | None = None
            for attempt in range(1, 4):
                try:
                    mg = bt.metagraph(netuid=netuid, network=args.network, lite=True, sync=True)
                    break
                except Exception as e:
                    logging.warning("Metagraph fetch failed (attempt %d/3): %s", attempt, e)
                    time.sleep(2)
            if mg is None:
                last_block = blk
                continue

            stats = cycle(netuid, blk, mg)
            try:
                committed = stats.get("committed", 0)
                valid = stats.get("valid", 0)
                pct_valid = (valid / committed * 100.0) if committed else 0.0
                logging.info(
                    "Block %s | payload quality: %.1f%% (%d/%d)",
                    blk,
                    pct_valid,
                    valid,
                    committed,
                )
            except Exception:
                pass

            if blk >= next_plaintext_update and (plaintext_update_task is None or not plaintext_update_task.is_alive()):
                def plaintext_worker(block_snapshot: int):
                    logging.info("=== Plaintext update start | block %s ===", block_snapshot)
                    pt.update_plaintext(miner_data)
                    logging.info("=== Plaintext update end | block %s ===", block_snapshot)

                plaintext_update_task = threading.Thread(target=plaintext_worker, args=(blk,), daemon=True)
                plaintext_update_task.start()
                logging.info("Plaintext update thread started for block %d.", blk)
                next_plaintext_update = blk + config.PLAINTEXT_UPDATE_INTERVAL

            if blk > 0 and blk % 30 == 0:
                logging.info("Saving plaintext...")
                pt.save_plaintext()
                logging.info("Plaintext saved.")

            if blk >= next_task and (weight_thread is None or not weight_thread.is_alive()):

                def worker(block_snapshot: int, uid_list, out_q: "queue.Queue[tuple[int, list[int], list[float]]]"):
                    weights_logger.info("=== Weight computation start | block %s ===", block_snapshot)
                    # pt.update_plaintext(miner_data) # Removed

                    try:
                        horizon = min(
                            len(rec["embeddings"])
                            for rec in plaintext_miner_data.values()
                            if isinstance(rec.get("embeddings"), list) and rec["embeddings"]
                        )
                    except ValueError:
                        horizon = 0

                    if horizon < config.LAG + 1:
                        weights_logger.info(
                            "Plaintext horizon too short (%d < %d) – deferring salience",
                            horizon,
                            config.LAG + 1,
                        )
                        return

                    min_hist_len = min(
                        (
                            len(r.get("history"))
                            for r in miner_data.values()
                            if isinstance(r.get("history"), list)
                        ),
                        default=0,
                    )
                    if min_hist_len < 2 * config.LAG + 1:
                        weights_logger.info(
                            "History too short (%d < %d) – deferring salience", 
                            min_hist_len,
                            2 * config.LAG + 1,
                        )
                        return

                    sal = pt.compute_salience_from_plaintext(sal_fn)
                    if not sal:
                        weights_logger.info("Salience unavailable – skipping at block %s", block_snapshot)
                        return

                    w = torch.tensor([sal[uid] if uid < len(sal) else 0.0 for uid in uid_list])
                    if w.sum() <= 0:
                        weights_logger.warning("Zero-sum weights at block %s – skip", block_snapshot)
                        return

                    w = (w / w.sum()).cpu().tolist()

                    out_q.put((block_snapshot, [int(u) for u in uid_list], w))
                    weights_logger.info(
                        "✓ Weights computed and queued at block %s (max=%.4f)",
                        block_snapshot,
                        max(w),
                    )
                    weights_logger.info("=== Weight computation end | block %s ===", block_snapshot)

                weight_thread = threading.Thread(target=worker, args=(blk, mg.uids, weight_queue), daemon=True)
                weight_thread.start()
                next_task = blk + config.TASK_INTERVAL

            # pt.update_plaintext(miner_data) # Removed

            if blk % 30 == 0:
                pt.save_plaintext()

            try:
                while True:
                    blk_snap, uid_list_queued, w_list = weight_queue.get_nowait()
                    w_tensor = torch.tensor(w_list, dtype=torch.float32)
                    try:
                        sub.set_weights(
                            netuid=netuid,
                            wallet=wallet,
                            uids=uid_list_queued,
                            weights=w_tensor,
                            wait_for_inclusion=False,
                        )
                        weights_logger.info(
                            "✓ set_weights submitted from queue (blk=%s, max=%.4f)",
                            blk_snap,
                            w_tensor.max(),
                        )
                    except Exception as e:
                        weights_logger.exception("Failed to set queued weights at block %s: %s", blk_snap, e)
                    finally:
                        weight_queue.task_done()
            except queue.Empty:
                pass

            last_block = blk
        time.sleep(2)


if __name__ == "__main__":
    main()

