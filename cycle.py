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

import asyncio, bittensor as bt, requests, config, comms

NETWORK = "finney"
sub = bt.subtensor(network=NETWORK)
miner_data: dict[int, dict] = {}

def cycle(netuid: int = 123, block: int = None, mg: bt.metagraph = None):
    ref = sub.get_timestamp(block)
    price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",timeout=2).json()["price"])

    
    commits = sub.get_all_commitments(netuid)
    print("commits", commits)

    uid2hot = dict(zip(mg.uids.tolist(), mg.hotkeys))
    print("uid2hot", uid2hot)
    zero = bt.timelock.encrypt(str([0]*config.FEATURE_LENGTH), n_blocks=1, block_time=1)[0]

    async def task(uid: int):
        hot = uid2hot.get(uid)
        prev = miner_data.get(uid, {})
        hist = prev.get("history", [])
        blocks = prev.get("blocks", [])
        btc = prev.get("btc", [])
        if prev.get("hotkey") != hot:
            hist = [zero] * len(hist)
            blocks = blocks[:len(hist)]
            btc = btc[:len(hist)]
        object_url = commits.get(hot) if hot else None
        payload = zero

        if object_url:
            try:
                from urllib.parse import urlparse

                path_parts = urlparse(object_url).path.lstrip("/").split("/")

                object_name = path_parts[-1] if path_parts else ""

                if object_name.lower() == (hot or "").lower():
                    ts = await comms.timestamp(object_url)
                    if ts is None or ts <= ref:
                        payload = await comms.download(object_url)
            except Exception as e:
                print(f"Error processing payload for uid {uid}: {e}")
                payload = zero
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

    async def run():
        await asyncio.gather(*(task(u) for u in range(config.NUM_UIDS)))

    try:
        asyncio.run(run())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(run())


