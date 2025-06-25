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

import asyncio, bittensor as bt, requests, config, comms, logging, os
from plaintext_utils import plaintext_miner_data, ZERO_VEC

logger = logging.getLogger(__name__)

NETWORK = "finney"
sub = bt.subtensor(network=NETWORK)
miner_data: dict[int, dict] = {}

MAX_PAYLOAD_BYTES = 25 * 1024 * 1024

ZERO_CIPHER = bt.timelock.encrypt(str([0] * config.FEATURE_LENGTH), n_blocks=1, block_time=1)[0]

def cycle(netuid: int = 123, block: int = None, mg: bt.metagraph = None):
    ref = sub.get_timestamp(block)
    price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",timeout=2).json()["price"])

    
    commits = sub.get_all_commitments(netuid)
    logger.debug("commits %s", commits)

    uid2hot = dict(zip(mg.uids.tolist(), mg.hotkeys))
    logger.debug("uid2hot mapping ready")
    zero = ZERO_CIPHER 

    async def task(uid: int):
        """Fetch payload for a single UID.
        For miners that have **never** committed, we leave ``history`` as ``None`` until
        their first valid commit arrives. When that happens, the entire backfill of
        zeros (encrypted) is generated in one shot so that their history length
        matches the current global length.
        """
        try:
            hot = uid2hot.get(uid)
            prev = miner_data.get(uid, {})

            hist = prev.get("history") 
            blocks = prev.get("blocks") or []
            btc = prev.get("btc") or []

            if isinstance(hist, list) and prev.get("hotkey") != hot:
                hist = [zero] * len(hist)
                blocks = blocks[: len(hist)]
                btc = btc[: len(hist)]

                try:
                    entry = plaintext_miner_data.get(uid)
                    if entry and isinstance(entry.get("embeddings"), list):
                        n = len(entry["embeddings"])
                        entry["embeddings"] = [ZERO_VEC] * n
                except Exception as e:
                    logger.warning("Failed to reset plaintext embeddings for UID %s after hotkey change: %s", uid, e)

            object_url = commits.get(hot) if hot else None


            if object_url is None and not isinstance(hist, list):
                miner_data[uid] = {
                    "uid": uid,
                    "hotkey": hot,
                    "history": None,
                    "object_url": None,
                    "blocks": None,
                    "btc": None,
                }
                return

            payload = zero 

            if object_url is not None:
                try:
                    from urllib.parse import urlparse

                    path_parts = urlparse(object_url).path.lstrip("/").split("/")
                    object_name = path_parts[-1] if path_parts else ""

                    if object_name.lower() == (hot or "").lower():
                        try:
                            payload_raw = await comms.download(object_url, max_size_bytes=MAX_PAYLOAD_BYTES)
                        except Exception as e:
                            logger.warning("Download failed for uid %s: %s", uid, e)
                            payload_raw = None

                        if payload_raw is not None:
                            payload = payload_raw
                        else:
                            logger.warning("Rejected payload for uid %s: invalid or too large", uid)
                except Exception as e:
                    logger.warning("Error processing payload for uid %s: %s", uid, e)

            if not isinstance(hist, list):
                current_len = next((len(r["history"]) for r in miner_data.values() if isinstance(r.get("history"), list)), 0)
                hist = [zero] * current_len
                blocks = [block] * current_len
                btc = [price] * current_len

            hist.append(payload)
            blocks.append(block)
            btc.append(price)

            miner_data[uid] = {
                "uid": uid,
                "hotkey": hot,
                "history": hist,
                "object_url": object_url,
                "blocks": blocks,
                "btc": btc,
            }
        except Exception as e:
            logger.exception("Unhandled error in task(uid=%s): %s", uid, e)
            miner_data[uid] = {
                "uid": uid,
                "hotkey": uid2hot.get(uid),
                "history": None,
                "object_url": None,
                "blocks": None,
                "btc": None,
            }

    async def run():
        await asyncio.gather(*(task(u) for u in range(config.NUM_UIDS)), return_exceptions=True)

        ref_len = 0
        for rec in miner_data.values():
            hist = rec.get("history")
            if isinstance(hist, list) and len(hist) > ref_len:
                ref_len = len(hist)

        for uid in range(config.NUM_UIDS):
            rec = miner_data.get(uid)
            if rec is None:
                miner_data[uid] = {
                    "uid": uid,
                    "hotkey": uid2hot.get(uid),
                    "history": None,
                    "object_url": None,
                    "blocks": None,
                    "btc": None,
                }
                continue

            hist = rec.get("history")
            if not isinstance(hist, list):
                continue

            while len(hist) < ref_len:
                hist.append(zero)
                rec["blocks"].append(block)
                rec["btc"].append(price)
        logger.debug("Cycle verification complete – all UIDs padded to length %s", ref_len)

    try:
        asyncio.run(run())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(run())


    committed = 0
    valid = 0
    for uid in range(config.NUM_UIDS):
        hot = uid2hot.get(uid)
        if commits.get(hot):
            committed += 1
            try:
                hist = miner_data[uid].get("history")
                if isinstance(hist, list) and hist and hist[-1] != ZERO_CIPHER:
                    valid += 1
            except Exception:
                pass

    return {"committed": committed, "valid": valid}


