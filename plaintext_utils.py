from __future__ import annotations

import ast
import gzip
import json
import os
import logging
from typing import Any, Dict, List
import time
import threading
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import bittensor as bt

import config

log = logging.getLogger("plaintext")

PLAINTEXT_PATH = os.path.expanduser("~/plaintext.data.gz")

plaintext_lock = threading.Lock()

plaintext_miner_data: Dict[int, Dict[str, List[Any]]] = {}


def load_plaintext() -> None:
    """Populate `plaintext_miner_data` from PLAINTEXT_PATH if it exists."""
    if not os.path.exists(PLAINTEXT_PATH):
        return
    try:
        with gzip.open(PLAINTEXT_PATH, "rb") as f:
            data = json.loads(f.read().decode())
        plaintext_miner_data.clear()
        for k, v in data.items():
            try:
                plaintext_miner_data[int(k)] = v
            except Exception:
                pass
        log.info("Loaded plaintext embeddings for %d UIDs", len(plaintext_miner_data))
    except Exception as e:
        log.warning("Failed to load plaintext archive: %s", e)


def save_plaintext() -> None:
    """Write current plaintext_miner_data to PLAINTEXT_PATH (gzip compressed)."""
    try:
        payload = json.dumps(plaintext_miner_data, separators=(",", ":")).encode()
        with gzip.open(PLAINTEXT_PATH + ".tmp", "wb", compresslevel=5) as f:
            f.write(payload)
        os.replace(PLAINTEXT_PATH + ".tmp", PLAINTEXT_PATH)
        log.info("✓ Saved plaintext archive (%d KB)", len(payload) // 1024)
    except Exception as e:
        log.warning("Failed to save plaintext archive: %s", e)


ZERO_VEC = [0.0] * config.FEATURE_LENGTH


def _dbytes(x):
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        try:
            return bytes.fromhex(x)
        except Exception:
            return None
    return None


def _hex_to_bytes(o):
    if not isinstance(o, dict):
        return
    for k, v in o.items():
        if isinstance(v, str) and v.startswith("0x"):
            try:
                o[k] = bytes.fromhex(v[2:])
            except ValueError:
                pass  # Not a valid hex string
        elif isinstance(v, dict):
            _hex_to_bytes(v)
        elif isinstance(v, list):
            for i in v:
                _hex_to_bytes(i)


def _decode(payload: Any) -> list[float]:
    """Return a list[float] of length FEATURE_LENGTH from raw payload.
    Falls back to zero vector on any error."""
    if isinstance(payload, list):
        return payload if len(payload) == config.FEATURE_LENGTH else ZERO_VEC
    try:
        if isinstance(payload, (bytes, bytearray)):
            val = bt.timelock.decrypt(payload)
            if isinstance(val, (bytes, bytearray)):
                val = val.decode()
            if isinstance(val, str):
                val = ast.literal_eval(val)
            if isinstance(val, list) and len(val) == config.FEATURE_LENGTH:
                return val
    except Exception:
        pass
    return ZERO_VEC


def _decrypt_new_entries(history, blocks, btc, start_index):
    """Worker function to decrypt history entries for a single UID."""
    new_embeddings = []
    new_blocks = []
    new_btc = []
    for i in range(start_index, len(history)):
        new_embeddings.append(_decode(history[i]))
        if blocks and i < len(blocks):
            new_blocks.append(blocks[i])
        if btc and i < len(btc):
            new_btc.append(btc[i])
    return new_embeddings, new_blocks, new_btc


def update_plaintext(miner_data: dict) -> None:
    """Decrypt new data from miner_data and update plaintext_miner_data in parallel."""
    with plaintext_lock:
        log = logging.getLogger("plaintext")
        log.info("--- Starting parallel plaintext update ---")
        t0 = time.time()
        
        decrypted_count = 0
        
        with ThreadPoolExecutor() as executor:
            future_to_uid = {}
            # First, prepare and submit tasks.
            for uid, data in miner_data.items():
                if not isinstance(data.get("history"), list):
                    continue
                
                # Ensure record exists and hotkey is up-to-date
                if uid not in plaintext_miner_data:
                    plaintext_miner_data[uid] = {
                        "uid": uid, 
                        "hotkey": data.get("hotkey"), 
                        "embeddings": [],
                        "blocks": [],
                        "btc": [],
                    }
                else:
                    plaintext_miner_data[uid]["hotkey"] = data.get("hotkey")
                
                pt_record = plaintext_miner_data[uid]
                num_processed = len(pt_record["embeddings"])
                history = data["history"]

                if len(history) > num_processed:
                    future = executor.submit(_decrypt_new_entries, 
                                             history, 
                                             data.get("blocks"), 
                                             data.get("btc"), 
                                             num_processed)
                    future_to_uid[future] = uid

            # Process results as they complete
            for future in as_completed(future_to_uid):
                uid = future_to_uid[future]
                try:
                    new_embeddings, new_blocks, new_btc = future.result()
                    
                    pt_record = plaintext_miner_data[uid]
                    pt_record["embeddings"].extend(new_embeddings)
                    pt_record["blocks"].extend(new_blocks)
                    pt_record["btc"].extend(new_btc)
                    
                    decrypted_count += len(new_embeddings)
                except Exception as exc:
                    log.error(f"UID {uid} generated an exception during parallel decryption: {exc}")

        log.info(f"--- Finished plaintext update. Decrypted {decrypted_count} new records in {time.time() - t0:.2f}s ---")


def compute_salience_from_plaintext(salience_fn, max_horizon: int | None = None) -> list[float] | None:
    """Prepare plaintext data and compute salience."""
    with plaintext_lock:
        # Deepcopy to avoid holding lock during long computation
        local_plaintext_data = copy.deepcopy(plaintext_miner_data)

    N = config.NUM_UIDS
    LAG = config.LAG

    lens = [len(rec["embeddings"]) for rec in local_plaintext_data.values() if isinstance(rec.get("embeddings"), list)]
    if not lens:
        return None

    max_T = max(lens)
    if max_T <= 0:
        return None

    ret_src = local_plaintext_data.get(1)
    if not ret_src or not isinstance(ret_src.get("returns"), list):
        return None

    btc_len = len(ret_src["returns"])
    T = min(max_T, btc_len)
    if T <= 0:
        return None

    history = defaultdict(list)
    returns = defaultdict(list)

    for uid in range(N):
        rec = local_plaintext_data.get(uid)
        if not rec or not isinstance(rec.get("embeddings"), list):
            continue
        history[uid] = rec["embeddings"][:T]
        if not rec or not isinstance(rec.get("btc"), list):
            continue
        
        btc_prices = rec["btc"]
        T_returns = min(len(btc_prices) - LAG, T)

    pct = ret_src["returns"][:T]

    return salience_fn(history, pct) 
