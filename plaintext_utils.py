from __future__ import annotations

import ast
import gzip
import json
import os
import logging
from typing import Any, Dict, List

import bittensor as bt

import config

log = logging.getLogger("plaintext")

PLAINTEXT_PATH = os.path.expanduser("~/plaintext.data.gz")

plaintext_miner_data: Dict[int, Dict[str, List[Any]]] = {}


def load_plaintext() -> None:
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


def update_plaintext(miner_data: Dict[int, Any]) -> None:

    N = config.NUM_UIDS
    D = config.FEATURE_LENGTH
    LAG = config.LAG

    active_lengths = [
        len(rec.get("history")) for rec in miner_data.values() if isinstance(rec.get("history"), list)
    ]
    if not active_lengths:
        return

    horizon = max(active_lengths)  

    for uid in range(N):
        hist = miner_data.get(uid, {}).get("history")
        btc = miner_data.get(uid, {}).get("btc")
        if not (isinstance(hist, list) and isinstance(btc, list)):
            continue

        entry = plaintext_miner_data.setdefault(uid, {"embeddings": [], "returns": []})
        emb_out = entry["embeddings"]
        ret_out = entry["returns"]

        start = len(emb_out)
        if start >= horizon:
            continue

        for t in range(start, horizon):
            enc = hist[t] if t < len(hist) else ZERO_VEC
            if isinstance(enc, list) and len(enc) == D:
                vec = enc
            else:
                raw = _dbytes(enc)
                vec = ZERO_VEC
                if raw:
                    try:
                        vec_candidate = bt.timelock.decrypt(raw)
                        if isinstance(vec_candidate, (bytes, bytearray)):
                            vec_candidate = vec_candidate.decode()
                        if isinstance(vec_candidate, str):
                            vec_candidate = ast.literal_eval(vec_candidate)
                        if isinstance(vec_candidate, list) and len(vec_candidate) == D:
                            vec = vec_candidate
                    except Exception:
                        pass
            emb_out.append(vec)
            try:
                if t < len(btc) and t + LAG < len(btc):
                    p, f = btc[t], btc[t + LAG]
                    r = (f - p) / p if p else 0.0
                else:
                    r = 0.0
            except Exception:
                r = 0.0
            ret_out.append(r)

    log.info("Plaintext update complete up to horizon %d (max mode, no truncation)", horizon)



def compute_salience_from_plaintext(sal_fn):
    """Return list[float] or None using already-decrypted embeddings.

    Parameters
    ----------
    sal_fn : callable
        Reference to model.salience function.
    """
    N = config.NUM_UIDS
    LAG = config.LAG

    lens = [len(rec["embeddings"]) for rec in plaintext_miner_data.values() if isinstance(rec.get("embeddings"), list)]
    if not lens:
        return None

    max_T = max(lens)
    if max_T <= 0:
        return None

    ret_src = plaintext_miner_data.get(1)
    if not ret_src or not isinstance(ret_src.get("returns"), list):
        return None

    btc_len = len(ret_src["returns"])
    T = min(max_T, btc_len)
    if T <= 0:
        return None

    history: Dict[int, List[List[float]]] = {}
    for uid in range(N):
        rec = plaintext_miner_data.get(uid)
        if rec and isinstance(rec.get("embeddings"), list):
            emb = rec["embeddings"]
            L = len(emb)
            if L >= T:
                history[uid] = emb[:T]
            else:
                pad_len = T - L
                history[uid] = [ZERO_VEC] * pad_len + emb
        else:
            history[uid] = [ZERO_VEC] * T

    pct = ret_src["returns"][:T]

    return sal_fn(history, pct) 
