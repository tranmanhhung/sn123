import asyncio, bittensor as bt, comms, requests

NETWORK = "finney"
sub = bt.subtensor(network=NETWORK)
miner_data: dict[int, dict] = {}

def cycle(netuid: int = 128, block: int = None, mg: bt.metagraph = None):
    ref = sub.get_timestamp(block)
    price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",timeout=2).json()["price"])

    
    commits = sub.get_all_commitments(netuid)
    uid2hot = dict(zip(mg.uids.tolist(), mg.hotkeys))
    zero = bt.timelock.encrypt(str([0]*100), n_blocks=1, block_time=1)[0]

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
        bucket = commits.get(hot) if hot else None
        payload = zero
        if bucket:
            keys = await comms.list(bucket, "")
            if len(keys) == 1:
                k = keys[0]
                ts = await comms.timestamp(bucket, k)
                if not ts or ts <= ref:
                    d = await comms.download(bucket, k)
                    if isinstance(d, list) and len(d) >= 2 and d[0] == hot:
                        payload = d[1]
        hist.append(payload)
        blocks.append(block)
        btc.append(price)
        miner_data[uid] = {"uid": uid, "hotkey": hot, "history": hist, "bucket": bucket, "blocks": blocks, "btc": btc}

    async def run():
        await asyncio.gather(*(task(u) for u in range(256)))

    try:
        asyncio.run(run())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(run())


