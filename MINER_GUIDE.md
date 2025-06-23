# MANTIS Mining Guide

A quick reference for setting up your miner.

> **Prerequisite:** Ensure your **hotkey is already registered on-chain** (i.e. it exists inside your Bittensor wallet and has been set up via `btcli` or an equivalent tool). Without a registered hotkey, the commit in step **BB** will fail.

---

## X. Build the 100-value embedding (`[-1, 1]`)
```python
import numpy as np
emb = np.random.uniform(-1, 1, size=100).tolist()  # replace with real model output
```

## Y. Timelock-encrypt the embedding
```python
import bittensor as bt
cipher_bytes = bt.timelock.encrypt(str(emb), n_blocks=1, block_time=1)[0]
```

## Z. Hex-encode & save to a file (file name == hotkey)
```python
hex_str = cipher_bytes.hex()
hotkey = "5DhoY..."              # your miner's hotkey / bucket name
with open(hotkey, "w") as f:
    f.write(hex_str)
```

## AA. Upload to Cloudflare R2 (bucket name == hotkey)
```python
from comms import upload
upload(bucket=hotkey, object_key=hotkey, file_path=hotkey)
```

## BB. Commit the public R2 URL on-chain
```python
import bittensor as bt
wallet = bt.wallet(name="subnet", hotkey="my_hotkey")
subtensor = bt.subtensor(network="finney")
url = f"https://pub-<hash>.r2.dev/{hotkey}"
subtensor.commit(wallet=wallet, netuid=123, data=url)
```

---

Flow: **(X)** create 100-value array → **(Y)** timelock → **(Z)** hex → **(AA)** upload → **(BB)** commit.
Do not commit every time, submit new arrays to the R2 bucket frequently.
**You are ready to go!** 
