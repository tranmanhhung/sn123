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

import time, torch, bittensor as bt, argparse, threading, logging, ast, requests, json, gzip, os
from cycle import cycle, miner_data
from model import salience as sal_fn
import config
from config import ARCHIVE_URL


def _deserialize_archive(data_bytes: bytes):
    """Return dict[int, Any] from raw bytes, accepting optional gzip wrapper."""
    try:
        # Detect gzip by magic number 0x1f8b
        if len(data_bytes) >= 2 and data_bytes[0] == 0x1F and data_bytes[1] == 0x8B:
            data_bytes = gzip.decompress(data_bytes)
    except Exception as e:
        logging.warning("Gzip decompression failed: %s", e)

    try:
        import orjson  # type: ignore
        return orjson.loads(data_bytes)
    except Exception:
        try:
            return json.loads(data_bytes.decode())
        except Exception as e:
            logging.error("Archive JSON parse failed: %s", e)
            raise

def compute_salience():
    N=config.NUM_UIDS
    LAG=config.LAG
    lengths=[len(miner_data[u]["history"]) for u in range(N)]
    T=min(lengths)-LAG
    if T<=0:
        return None
    def dbytes(x):
        if isinstance(x,(bytes,bytearray)):return bytes(x)
        if isinstance(x,str):
            try:return bytes.fromhex(x)
            except:pass
        return None


    D=config.FEATURE_LENGTH
    dec={u:[] for u in range(N)}
    for t in range(T):
        for u in range(N):
            enc=miner_data[u]["history"][t]
            raw=dbytes(enc)
            try:
                if not raw:
                    raise ValueError("Empty payload")

                try:
                    v = bt.timelock.decrypt(raw)
                except Exception as e:
                    raise ValueError(f"Timelock decrypt failed: {e}") from e

                if isinstance(v, (bytes, bytearray)):
                    try:
                        v = v.decode()
                    except Exception as e:
                        raise ValueError(f"Byte->str decode failed: {e}") from e

                if isinstance(v, str):
                    try:
                        v = ast.literal_eval(v)
                    except Exception as e:
                        raise ValueError(f"literal_eval failed: {e}") from e

                if not (isinstance(v, list) and len(v) == D):
                    raise ValueError(f"Expected list of length {D}, got type {type(v)} len {len(v) if isinstance(v,list) else 'N/A'}")

                if any((not isinstance(x, (int, float)) or x < -1 or x > 1) for x in v):
                    raise ValueError("Values out of [-1,1] range or non-numeric present")

            except Exception as e:
                logging.warning(f'Invalid embedding at uid {u} timestep {t}: {e}. Using zeros.')
                v = [0] * D
            dec[u].append(v)
    btc=miner_data[0]["btc"]
    pct=[0.0]*T
    for i in range(T):
        p=btc[i]; f=btc[i+LAG]
        if p and f and p!=0:
            pct[i]=(f-p)/p
    print("salience computed")
    return sal_fn(dec, pct)



def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wallet.name",required=True)
    p.add_argument("--wallet.hotkey",required=True)
    p.add_argument("--network",default="finney")
    p.add_argument("--netuid",type=int,default=123)
    args=p.parse_args()

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    sub=bt.subtensor(network=args.network)
    wallet=bt.wallet(name=getattr(args, 'wallet.name'),hotkey=getattr(args, 'wallet.hotkey'))


    try:
        logging.info("Fetching initial miner_data archive from %s", ARCHIVE_URL)
        resp = requests.get(ARCHIVE_URL, timeout=30)
        resp.raise_for_status()
        data_bytes = resp.content
        data = _deserialize_archive(data_bytes)
        miner_data.clear()
        for k, v in data.items():
            try:
                miner_data[int(k)] = v
            except Exception:
                continue
        logging.info("Loaded %d UIDs from archive", len(miner_data))
        if miner_data:
            length = min(len(rec.get("history", [])) for rec in miner_data.values())
            logging.info("Archive timestep length: %d", length)
    except Exception as e:
        logging.warning("Could not initialise miner_data from archive: %s", e)

    netuid=args.netuid
    last=sub.get_current_block()
    next_task=last+config.TASK_INTERVAL
    task=None
    while True:
        b=sub.get_current_block()
        if b!=last:
            print("new block")
            mg=bt.metagraph(netuid=netuid,network=args.network,lite=True,sync=True)
            b=sub.get_current_block()
            cycle(netuid,b,mg)

            if b>=next_task and (task is None or not task.is_alive()):
                def worker(block_snapshot,uid_list):
                    sal=compute_salience()
                    if sal:
                        w=torch.tensor([sal[uid] if uid<len(sal) else 0.0 for uid in uid_list])
                        if w.sum()>0:
                            w=w/w.sum()
                            print("setting weights",w)
                            sub.set_weights(netuid=netuid,wallet=wallet,uids=uid_list,weights=w,wait_for_inclusion=False)
                task=threading.Thread(target=worker,args=(b,mg.uids),daemon=True)
                task.start()
                next_task=b+config.TASK_INTERVAL
            last=b
        time.sleep(2)

if __name__=="__main__":
    main()

