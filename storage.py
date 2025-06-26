import gzip
import json
import logging
import os
import pickle
import ast
from typing import Any, Dict, List
import requests
import asyncio
import time
import secrets
import aiohttp

from timelock import Timelock

import bittensor as bt
import torch

import config

logger = logging.getLogger(__name__)

# --- tlock/Drand Configuration ---
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

# --- Constants ---
ZERO_VEC = [0.0] * config.FEATURE_LENGTH

def _precompute_encrypted_zero():
    """Creates a pre-computed tlock payload for the zero vector."""
    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        # Use a fixed, past round that is guaranteed to have a signature.
        round_num = 1
        vector_str = str(ZERO_VEC)
        salt = secrets.token_bytes(32)
        
        ciphertext_hex = tlock.tle(round_num, vector_str, salt).hex()
        
        payload_dict = {"round": round_num, "ciphertext": ciphertext_hex}
        return json.dumps(payload_dict).encode("utf-8")
    except Exception as e:
        logger.error(f"Failed to pre-compute encrypted zero vector, using fallback: {e}")
        # This fallback will be safely handled as a parsing failure later.
        return b'{"round": 1, "ciphertext": "error"}'

ENCRYPTED_ZERO_PAYLOAD = _precompute_encrypted_zero()


class DataLog:
    """
    A unified, append-only log for all historical data in the subnet.

    This class manages the complete state of miner data, including block numbers,
    BTC prices, raw encrypted payloads, and a cache for decrypted plaintext data.
    It is designed to be the single source of truth, ensuring data integrity and
    consistent history length across all miners.

    The log is persisted to a single file, making it a self-contained and
    portable data store.
    """

    def __init__(self):
        # The block number for each timestep.
        self.blocks: List[int] = []
        # The BTC price at the time of each block.
        self.btc_prices: List[float] = []
        # A dense cache of decrypted plaintext data. Maps timestep -> uid -> vector.
        self.plaintext_cache: List[Dict[int, List[float]]] = []
        # A sparse dictionary of raw, unprocessed payloads.
        # Maps timestep -> uid -> encrypted_bytes.
        # This is a "to-do" list; payloads are removed after processing.
        self.raw_payloads: Dict[int, Dict[int, bytes]] = {}

        # Initialize tlock and cache for Drand info
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        self._drand_info: Dict[str, Any] = {}
        self._drand_info_last_update: float = 0

    async def _get_drand_info(self) -> Dict[str, Any]:
        """Gets and caches Drand network info."""
        if not self._drand_info or time.time() - self._drand_info_last_update > 3600:
            try:
                url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        response.raise_for_status()
                        self._drand_info = await response.json()
                self._drand_info_last_update = time.time()
                logger.info("Updated Drand beacon info.")
            except Exception as e:
                logger.error(f"Failed to get Drand info: {e}")
                return {}
        return self._drand_info

    async def _get_drand_signature(self, round_num: int) -> bytes | None:
        """Fetches the signature for a specific Drand round."""
        # Add a small buffer to give the beacon time to publish.
        await asyncio.sleep(2)
        try:
            url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"✅ Signature fetched for round {round_num}")
                        return bytes.fromhex(data["signature"])
                    else:
                        logger.warning(
                            f"-> Failed to fetch signature for round {round_num}, "
                            f"status: {response.status}"
                        )
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"-> Timeout fetching signature for round {round_num}")
            return None
        except Exception as e:
            logger.error(f"-> Error fetching signature for round {round_num}: {e}")
            return None

    def append_step(
        self, block: int, btc_price: float, payloads: Dict[int, bytes]
    ) -> None:
        """Appends a new timestep to the log."""
        self.blocks.append(block)
        self.btc_prices.append(btc_price)
        self.plaintext_cache.append({})  # Add a new, empty dict for this timestep.

        current_timestep = len(self.blocks) - 1
        self.raw_payloads[current_timestep] = {}

        # Ensure all known miners have a default entry for this step if missing.
        all_known_uids = self.get_all_uids()
        for uid in all_known_uids:
            if uid not in payloads:
                payloads[uid] = ENCRYPTED_ZERO_PAYLOAD

        for uid, payload in payloads.items():
            self.raw_payloads[current_timestep][uid] = payload

            # If this is a new miner, backfill their entire history with zeros.
            if not self.is_known_uid(uid):
                self._backfill_new_uid(uid)

    def get_all_uids(self) -> List[int]:
        """Returns a sorted list of all unique UIDs ever seen in the log."""
        uids = set()
        for step_cache in self.plaintext_cache:
            uids.update(step_cache.keys())
        for step_payloads in self.raw_payloads.values():
            uids.update(step_payloads.keys())
        return sorted(list(uids))

    def is_known_uid(self, uid: int) -> bool:
        """Check if a UID has any data in the first timestep."""
        if not self.plaintext_cache:
            return False
        return uid in self.plaintext_cache[0]

    def _backfill_new_uid(self, uid: int) -> None:
        """Fills the history for a new UID with default zero values."""
        # There is no history to backfill if this is the first or second entry.
        # The loop below will be empty anyway, this just prevents log spam.
        if len(self.plaintext_cache) <= 1:
            return

        logger.info(f"🚀 New miner detected (UID: {uid}). Backfilling history.")
        # Iterate up to the new, current timestep, but do not fill it.
        # It will be filled later by process_pending_payloads.
        for step_cache in self.plaintext_cache[:-1]:
            if uid not in step_cache:
                step_cache[uid] = ZERO_VEC

    async def process_pending_payloads(self) -> None:
        """
        Groups pending payloads by Drand round, fetches signatures for ready
        rounds, and decrypts them in batches.
        """
        current_block = self.blocks[-1] if self.blocks else 0
        rounds_to_process: Dict[int, List[Dict]] = {}
        processed_payload_keys: List[tuple[int, int]] = []

        # 1. Group payloads by Drand round
        # Iterate over a copy of the items to prevent mutation issues during the loop.
        for ts, payloads_at_step in list(self.raw_payloads.items()):
            block_age = current_block - self.blocks[ts]
            if not (61 <= block_age <= 120):
                if block_age > 120:
                    logger.warning(f"Discarding stale raw payloads at timestep {ts}")
                    processed_payload_keys.extend([(ts, u) for u in payloads_at_step.keys()])
                continue

            for uid, payload_bytes in list(payloads_at_step.items()):
                try:
                    # Ensure we are working with bytes
                    if isinstance(payload_bytes, dict):
                        # This is a corrupted entry from a previous bug, convert it back for processing
                        p = payload_bytes
                    else:
                        p = json.loads(payload_bytes)

                    round_num = p["round"]
                    if round_num not in rounds_to_process:
                        rounds_to_process[round_num] = []
                    rounds_to_process[round_num].append(
                        {"ts": ts, "uid": uid, "ct_hex": p["ciphertext"]}
                    )
                except Exception:
                    # This handles parsing failures (e.g., from the fallback payload)
                    self.plaintext_cache[ts][uid] = ZERO_VEC
                    processed_payload_keys.append((ts, uid))
        
        # 2. Fetch signatures and decrypt ready rounds
        for round_num, items in rounds_to_process.items():
            sig = await self._get_drand_signature(round_num)
            if not sig:
                continue  # Signature not yet available

            logger.info(f"Decrypting batch of {len(items)} payloads for Drand round {round_num}")

            # Batch decrypt all ciphertexts for this round
            for item in items:
                ts, uid, ct_hex = item["ts"], item["uid"], item["ct_hex"]
                try:
                    pt_bytes = self.tlock.tld(bytes.fromhex(ct_hex), sig)
                    vector = ast.literal_eval(pt_bytes.decode())
                    if self._validate_vector(vector):
                        self.plaintext_cache[ts][uid] = vector
                    else:
                        self.plaintext_cache[ts][uid] = ZERO_VEC
                except Exception as e:
                    logger.error(f"tlock decryption failed for UID {uid} at ts {ts}: {e}")
                    self.plaintext_cache[ts][uid] = ZERO_VEC
                finally:
                    processed_payload_keys.append((ts, uid))
        
        # 3. Clean up processed payloads from the raw queue
        for ts, uid in processed_payload_keys:
            if ts in self.raw_payloads and uid in self.raw_payloads[ts]:
                del self.raw_payloads[ts][uid]
                if not self.raw_payloads[ts]:
                    del self.raw_payloads[ts]

    def get_training_data(self) -> tuple[dict[int, list], list[float]] | None:
        """
        Prepares and returns the data required for training the salience model.

        This function reads directly from the `plaintext_cache` and calculates
        BTC price returns based on the `LAG` constant. It ensures all data is
        aligned and correctly formatted for the model.
        """
        if not self.plaintext_cache or len(self.blocks) < config.LAG * 2 + 1:
            logger.warning("Not enough data to create a training set.")
            return None

        T = len(self.blocks)
        all_uids = self.get_all_uids()
        history_dict = {uid: [] for uid in all_uids}
        btc_returns = []

        for t in range(T - config.LAG):
            p_initial = self.btc_prices[t]
            p_final = self.btc_prices[t + config.LAG]
            if p_initial > 0:
                btc_returns.append((p_final - p_initial) / p_initial)
            else:
                btc_returns.append(0.0)

        effective_T = len(btc_returns)
        for t in range(effective_T):
            for uid in all_uids:
                vector = self.plaintext_cache[t].get(uid, ZERO_VEC)
                history_dict[uid].append(vector)

        return history_dict, btc_returns

    @staticmethod
    def _validate_vector(vector: Any) -> bool:
        """Checks if a decrypted vector is valid."""
        if not isinstance(vector, list) or len(vector) != config.FEATURE_LENGTH:
            return False
        return all(isinstance(v, (int, float)) and -1.0 <= v <= 1.0 for v in vector)

    def save(self, path: str) -> None:
        """Saves the entire DataLog object to a compressed file."""
        try:
            tmp_path = path + ".tmp"
            with gzip.open(tmp_path, "wb") as f:
                pickle.dump(self, f)
            os.replace(tmp_path, path)
            logger.info(f"💾 Saved DataLog to {path}")
        except Exception as e:
            logger.error(f"Error saving DataLog: {e}")

    @staticmethod
    def load(path: str) -> "DataLog":
        """
        Loads a DataLog object.

        It first tries to load from the local `path`. If the file doesn't exist,
        it attempts to download it from the `DATALOG_ARCHIVE_URL` in the config.
        If both fail, it returns a new, empty DataLog.
        """
        if os.path.exists(path):
            logger.info(f"Found local DataLog at {path}")
            try:
                with gzip.open(path, "rb") as f:
                    log = pickle.load(f)
                logger.info(f"✅ Loaded DataLog from {path}")
                return log
            except Exception as e:
                logger.error(f"Failed to load local DataLog from {path}: {e}")
                logger.warning("Starting with a new, empty DataLog.")
                return DataLog()

        logger.warning(f"No local DataLog found. Attempting to download from archive...")
        try:
            url = config.DATALOG_ARCHIVE_URL
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"✅ Downloaded and saved archive to {path}")
            return DataLog.load(path)
        except Exception as e:
            logger.error(f"Failed to download or load archive from {config.DATALOG_ARCHIVE_URL}: {e}")
            logger.warning("Starting with a new, empty DataLog.")
            return DataLog() 
