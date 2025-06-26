# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MANTIS is a Bittensor subnet validator (subnet 123) that evaluates miner-submitted embeddings for Bitcoin price prediction using machine learning salience analysis. The system uses timelock encryption and a commit-reveal scheme to ensure fair evaluation.

## Key Architecture Components

### Data Flow
1. **Data Collection** (`cycle.py`): Fetches Bitcoin prices and encrypted miner embeddings from R2 storage
2. **Salience Analysis** (`model.py`): Uses neural networks to compute predictive importance of each miner
3. **Weight Setting** (`validator.py`): Assigns network weights based on salience scores every 360 blocks

### Core Configuration (`config.py`)
- `NETUID = 123` - Bittensor subnet identifier
- `FEATURE_LENGTH = 100` - Embedding vector dimension
- `LAG = 300` - Time lag for Bitcoin price prediction (blocks)
- `TASK_INTERVAL = 360` - Frequency of weight updates (blocks)

### Global State Management
- `miner_data` dict in `cycle.py` maintains historical data for all 256 UIDs
- Data structure: `{uid: {"history": [], "blocks": [], "btc": [], "hotkey": str, "bucket": str}}`
- History resets when miner hotkeys change (security measure)

## Running the Validator

```bash
python validator.py --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --network finney --netuid 123
```

Required environment variables for R2 storage:
- `R2_BUCKET_ID`
- `R2_ACCOUNT_ID` 
- `R2_WRITE_ACCESS_KEY_ID`
- `R2_WRITE_SECRET_ACCESS_KEY`

## Development Notes

### Testing
No specific test framework configured. The codebase includes inline validation and error handling.

### Key Dependencies
- `bittensor` - Blockchain integration and timelock encryption
- `torch` - Neural network implementation
- `aiobotocore` - Async R2 storage operations
- `requests` - Bitcoin price API calls

### Timelock Encryption Flow
1. Miners submit timelock-encrypted embeddings to R2 buckets
2. Validator downloads encrypted data during collection cycle
3. During evaluation, validator decrypts historical data using `bt.timelock.decrypt()`
4. Malformed/missing data defaults to zero vectors

### Salience Computation
The ML evaluation works by:
1. Training baseline model on all miner embeddings to predict Bitcoin returns
2. For each UID, retraining model with that UID's features masked to zero
3. Computing performance degradation (delta loss) when each miner is removed
4. Normalizing delta losses to create weight distribution

### Error Handling Patterns
- Network operations have timeout protections
- Missing/corrupted data gracefully defaults to zero embeddings
- Async operations use `asyncio.gather()` for parallel processing
- Logging at INFO level for main operations, DEBUG for detailed metrics