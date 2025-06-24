
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

# Structure: plaintext_miner_data[uid] = {"embeddings": list[list[float]], "returns": list[float]}
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


def update_plaintext(miner_data: Dict[int, Any]) -> None:
    """Ensure plaintext_miner_data contains decrypted data for every UID up to
    `horizon = min_hist_len - config.LAG`.
    """
    N = config.NUM_UIDS
    D = config.FEATURE_LENGTH
    LAG = config.LAG

    active_lengths = [
        len(rec.get("history")) for rec in miner_data.values() if isinstance(rec.get("history"), list)
    ]
    if not active_lengths:
        return
    horizon = min(active_lengths) - LAG
    if horizon <= 0:
        return

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
            enc = hist[t]
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
                p, f = btc[t], btc[t + LAG]
                r = (f - p) / p if p else 0.0
            except Exception:
                r = 0.0
            ret_out.append(r)

        if len(emb_out) > horizon:
            emb_out[:] = emb_out[:horizon]
            ret_out[:] = ret_out[:horizon]

    log.info("Plaintext update complete up to horizon %d", horizon)



def compute_salience_from_plaintext(sal_fn):
    """Return list[float] or None using already-decrypted embeddings.

    Parameters
    ----------
    sal_fn : callable
        Reference to model.salience function.
    """
    N = config.NUM_UIDS
    LAG = config.LAG

    horizons = []
    for uid in range(N):
        rec = plaintext_miner_data.get(uid)
        if rec and isinstance(rec.get("embeddings"), list):
            horizons.append(len(rec["embeddings"]))
    if not horizons:
        return None

    T = min(horizons)
    if T <= 0:
        return None

    history: Dict[int, List[List[float]] | None] = {}
    for uid in range(N):
        rec = plaintext_miner_data.get(uid)
        if rec and isinstance(rec.get("embeddings"), list) and len(rec["embeddings"]) >= T:
            history[uid] = rec["embeddings"][:T]
        else:
            history[uid] = None 

    ret_src = plaintext_miner_data.get(1)
    if not ret_src or len(ret_src.get("returns", [])) < T:
        return None

    pct = ret_src["returns"][:T]

    return sal_fn(history, pct) 
