# MANTIS Bittensor Subnet Validator – Codebase Overview

## System Architecture

This repository implements a fully–featured Bittensor **validator** (subnet 123) that evaluates miners who broadcast time-locked Bitcoin-price embeddings. The validator performs four high-level tasks:

1. **Collect** encrypted embeddings from miners and the reference Bitcoin spot price for every new block.
2. **Archive & cache** the raw data via Cloudflare R2 to guarantee transparency and reproducibility.
3. **Estimate salience** – i.e. the predictive contribution of each miner – with an online proxy-model that is trained on-chain.
4. **Set network weights** proportional to salience so that the most useful miners receive the largest staking rewards.

## Core Components Overview

### 1. Configuration Management (`config.py`)
The single source-of-truth for tunable hyper-parameters.

| Constant | Purpose |
|----------|---------|
| `ARCHIVE_URL` | Public URL hosting the latest `miner_data` snapshot. |
| `NETUID` | Identifier of the Bittensor subnet (defaults to **123**). |
| `NUM_UIDS` | Maximum number of permitted miner UIDs. |
| `FEATURE_LENGTH` | Embedding vector dimensionality. |
| `HIDDEN_SIZE` | Hidden layer width for the proxy MLP used during salience estimation. |
| `LEARNING_RATE` | Learning rate for the proxy model optimiser. |
| `LAG` | Prediction horizon (in blocks) when computing BTC percentage-change. |
| `TASK_INTERVAL` | Number of blocks between successive weight-setting events. |

### 2. Cloud Storage Communication (`comms.py`)
Handles **all** interaction with Cloudflare R2:

#### Configuration Helpers
- `bucket()` – Returns the active R2 bucket identifier from `R2_BUCKET_ID`.
- `load_r2_account_id()` – Reads Cloudflare account ID from the environment.
- `load_r2_endpoint_url()` – Builds the R2 S3-compatible endpoint URL.
- `load_r2_write_access_key_id()` / `load_r2_write_secret_access_key()` – Credentials for signed writes.

#### Local-Cache Utilities
- `get_local_path(bucket, filename)` – Maps an object key to `~/storage/<bucket>/<file>`.
- `exists_locally(bucket, filename)` – Async check for cached files.
- `delete_locally(bucket, filename)` – Async removal of a cached file.
- `load(bucket, filename)` – Async JSON loader from local disk.

#### Remote-Object Operations
- `_local_path_from_url(url)` – Internal helper underpinning the HTTP cache.
- `download(url)` – Fetches public objects (JSON / text / bytes) **and** persists them locally.
- `exists(bucket, filename)` – Async S3 HEAD request to test object existence.
- `timestamp(bucket, filename)` – Last-modified timestamp for a private R2 object.
- `list(bucket, prefix)` – Glob-style listing of keys under a prefix.
- `timestamp(url)` – (HTTP version) Last-Modified header scraper for public objects.
- `_sanitize_b64(obj)` – Recursively base-64 encodes arbitrary binary blobs.
- `upload(bucket, object_key, file_path)` – High-level wrapper that performs a signed PUT of a local file to R2.

> **Note** `new_comms.py` offers a _minimal_ read-only subset (`download`, `timestamp`) for consumer-only use-cases.

### 3. Data Collection Cycle (`cycle.py`)
Continuously mirrors on-chain commitments into an in-memory Python structure called **`miner_data`**.

Global state:
```
miner_data: Dict[int, {
    uid:      int,
    hotkey:   str | None,
    history:  List[bytes | str],   # encrypted embeddings
    object_url: str | None,        # R2 link advertised on-chain
    blocks:   List[int],           # block-heights when data was seen
    btc:      List[float],         # spot price at the same height
}]
```

Key routine:
- `cycle(netuid, block, mg)`
  1. Fetches the Binance BTC/USDT spot price.
  2. Pulls the latest **commitments** (`subtensor.get_all_commitments`).
  3. Builds a `uid → hotkey` lookup from the provided metagraph.
  4. Downloads each miner's advertised object via `comms.download()` (size-capped to 25 MB).
  5. Resets history if the miner rotates their hotkey (sybil defence).
  6. Appends `(ciphertext, block, price)` to global `miner_data`.

### 4. Machine Learning Model (`model.py`)
Provides the lightweight proxy model used to quantify salience.

#### `MLP` Class
| Layer | Details |
|-------|---------|
| `Linear(input → hidden)` | Width = `HIDDEN_SIZE`; ReLU |
| `Linear(hidden → hidden)` | Width = `HIDDEN_SIZE`; ReLU |
| `Linear(hidden → 1)` | Output; **Tanh** activation |

#### `salience(history_dict, btc_prices, …)`
1. Builds a flattened feature matrix of shape `(T, NUM_UIDS × FEATURE_LENGTH)`.
2. Trains the proxy MLP online using a **lagged** supervision scheme.
3. For each UID, re-runs the forward pass with that UID's slice zeroed out to measure the increase in loss.
4. Normalises positive delta-losses into a probability distribution → salience weights.

### 5. Salience Evaluation Loop (`main.py`)
Top-level orchestrator that bridges the on-chain world with the ML pipeline.

Important functions:
- `compute_salience()` –
  * Validates that all miners share a common history length ≥ `LAG`.
  * Decrypts embeddings via `bt.timelock.decrypt` and performs exhaustive sanity checks.
  * Calls `model.salience()` → returns a list of salience scores.
- `main()` – CLI entry-point which:
  1. Parses wallet / network flags.
  2. Pre-loads `miner_data` from the public `ARCHIVE_URL` snapshot.
  3. Enters an **infinite loop**:
     - On every new block: calls `cycle()` to ingest fresh data.
     - Every `TASK_INTERVAL` blocks: spawns a background thread that
       i) recomputes salience, ii) normalises to `torch.Tensor`, and
       iii) submits weights via `subtensor.set_weights()`.

### 6. Archive Decoder Utility (`decode.py`)
Small CLI helper to _inspect_ the public miner-data archive:
- `load_archive(url)` – Downloads the JSON snapshot and **re-casts** UID keys from strings → ints.
- `main()` – Prints the timestep length and per-UID embedding counts for quick sanity checking.

---

## System Workflow

1. **Initialisation** – Validator launches, restores `miner_data` from the last published archive (if available) and connects to the Subtensor network.
2. **Collection (every block)** – `cycle()` harvests commitments and BTC price, updating the in-memory dataset.
3. **Evaluation (every `TASK_INTERVAL` blocks)** – Background thread decrypts embeddings, computes salience, and emits weights.
4. **Reward Distribution** – `subtensor.set_weights()` finalises the weight vector on-chain; miners get paid proportionally.

## Security Considerations

- **Time-lock encryption** ensures miners cannot cheat by encoding future information.
- **Hotkey continuity** – History is reset when a UID changes hotkey, guarding against identity shuffle attacks.
- **Payload size limits** stop denial-of-service via oversized objects.
- **Environment isolation** – All R2 credentials are pulled from env-vars; none are hard-coded.

## Dependencies

- `bittensor` – Blockchain client & timelock cryptography.
- `torch` – Neural-network backbone.
- `requests`, `aiohttp` – HTTP I/O (synchronous & asynchronous).
- `aiobotocore`, `boto3` – Cloudflare R2 S3 compatibility.
- `numpy`, `pandas` – Numeric utilities.

## Performance Notes

- **Feature matrix assembly** – O(`NUM_UIDS × T × FEATURE_LENGTH`).
- **Proxy-model training** – Approximately O(`T × HIDDEN_SIZE × NUM_UIDS`).
- **Asynchronous I/O** keeps data-collection latency negligible relative to block times.

## Extensibility

The architecture is intentionally modular:

- Swap-in alternative salience algorithms by editing `model.py` only.
- Use a different storage backend by replacing the functions in `comms.py`.
- Increase embedding dimensionality by changing `FEATURE_LENGTH` – the model adapts automatically.

---

© 2024 MANTIS – Released under the MIT License.
