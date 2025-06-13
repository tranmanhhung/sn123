import time, torch, bittensor as bt, argparse, threading, logging, ast
from cycle import cycle, miner_data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wallet_name",required=True)
    p.add_argument("--hotkey_name",required=True)
    p.add_argument("--axon_port",type=int,default=8091)
    p.add_argument("--netuid",type=int,default=128)
    p.add_argument("--network",default="finney")
    args=p.parse_args()

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    sub=bt.subtensor(network=args.network)
    wallet=bt.wallet(name=args.wallet_name,hotkey=args.hotkey_name)
    bt.axon(wallet=wallet,port=args.axon_port).start()

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

import torch, bittensor as bt
from model import salience as sal_fn

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
