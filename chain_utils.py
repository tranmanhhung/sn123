import bittensor as bt


sub = bt.subtensor(network="finney")

def commit_r2_bucket(bucket: str, wallet, hotkey, uid, subtensor):
    try:
        result = subtensor.commit(wallet, uid, bucket, 360)
        assert result, "subtensor.commit did not return True"
        return result
    except Exception as e:
        print(f"Error committing bucket: {e}")
        return False
