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

import time, torch, bittensor as bt, argparse, threading, logging, ast; from cycle import cycle, miner_data; from model import salience as sal_fn


def commit_r2_bucket(bucket: str, wallet, hotkey, uid, subtensor):
    try:
        result = subtensor.commit(wallet, uid, bucket, 360)
        assert result, "subtensor.commit did not return True"
        return result
    except Exception as e:
        print(f"Error committing bucket: {e}")
        return False




def compute_salience():
    N=256
    LAG=300
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

    first=dbytes(miner_data[0]["history"][0])
    try:
        sample=bt.timelock.decrypt(first) if first else None
        if isinstance(sample,(bytes,bytearray)):sample=sample.decode()
        D=100
    except:
        logging.warning('Failed to infer embedding dimension, defaulting to 100')
        D=100
    dec={u:[] for u in range(N)}
    for t in range(T):
        for u in range(N):
            enc=miner_data[u]["history"][t]
            raw=dbytes(enc)
            try:
                v=bt.timelock.decrypt(raw) if raw else [0]*D
                if isinstance(v,(bytes,bytearray)):
                    v=v.decode()
                if isinstance(v,str):
                    v=ast.literal_eval(v)
                if not (isinstance(v,list) and len(v)==D):
                    raise ValueError
            except:
                logging.warning(f'Invalid embedding, using zeros at uid {u} timestep {t}')
                v=[0]*D
            dec[u].append(v)
    btc=miner_data[0]["btc"]
    pct=[0.0]*T
    for i in range(T):
        p=btc[i]; f=btc[i+LAG]
        if p and f and p!=0:
            pct[i]=(f-p)/p
    return sal_fn(dec, pct)



def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wallet_name",required=True)
    p.add_argument("--hotkey_name",required=True)
    p.add_argument("--network",default="finney")
    p.add_argument("--netuid",type=int,default=1)
    args=p.parse_args()

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    sub=bt.subtensor(network=args.network)
    wallet=bt.wallet(name=args.wallet_name,hotkey=args.hotkey_name)

    netuid=args.netuid
    last=sub.block()
    next_task=last+360
    task=None
    while True:
        b=sub.block()
        if b!=last:
            mg=bt.metagraph(netuid=netuid,network=args.network,lite=True,sync=True)
            b=sub.block()
            cycle(netuid,b,mg)

            if b>=next_task and (task is None or not task.is_alive()):
                def worker(block_snapshot,uid_list):
                    sal=compute_salience()
                    if sal:
                        w=torch.tensor([sal[uid] if uid<len(sal) else 0.0 for uid in uid_list])
                        if w.sum()>0:
                            w=w/w.sum()
                            sub.set_weights(netuid=netuid,wallet=wallet,uids=uid_list,weights=w,wait_for_inclusion=False)
                task=threading.Thread(target=worker,args=(b,mg.uids),daemon=True)
                task.start()
                next_task=b+360
            last=b
        time.sleep(2)

if __name__=="__main__":
    main()

