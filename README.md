# MANTIS Bittensor Subnet Validator - Codebase Overview

## System Architecture

This is a sophisticated Bittensor subnet validator (subnet 123) that implements a machine learning-based evaluation system for miner embeddings. The validator uses a commit-reveal scheme to collect encrypted embeddings from miners, evaluates their predictive power for Bitcoin price movements, and assigns network weights based on salience analysis.

## Core Components Overview

### 1. Configuration Management (config.py)
- **NETUID**: 123 - The Bittensor subnet identifier
- **NUM_UIDS**: 256 - Maximum number of UIDs in the network
- **FEATURE_LENGTH**: 100 - Dimensionality of embedding vectors
- **HIDDEN_SIZE**: 64 - Neural network hidden layer size
- **LEARNING_RATE**: 1e-3 - Training learning rate
- **LAG**: 300 - Time lag for Bitcoin price prediction (blocks)
- **TASK_INTERVAL**: 360 - Frequency of weight updates (blocks)

### 2. Cloud Storage Communication (comms.py)
Manages asynchronous data exchange with Cloudflare R2 storage buckets.

#### Configuration Functions:
- **bucket()**: Retrieves R2 bucket ID from environment variables
- **load_r2_account_id()**: Gets Cloudflare account ID
- **load_r2_endpoint_url()**: Constructs R2 endpoint URL
- **load_r2_write_access_key_id()**: Retrieves write access key
- **load_r2_write_secret_access_key()**: Retrieves secret access key

#### File Management Functions:
- **get_local_path(bucket, filename)**: Computes local storage path in ~/storage/
- **exists_locally(bucket, filename)**: Checks if file exists in local cache
- **delete_locally(bucket, filename)**: Removes file from local storage
- **load(bucket, filename)**: Loads JSON data from local file system
- **download(bucket, filename)**: Downloads file from R2 and caches locally
- **upload(bucket, filename, data)**: Uploads JSON data to R2 and saves locally
- **exists(bucket, filename)**: Checks if file exists on R2 storage
- **timestamp(bucket, filename)**: Gets last modification time from R2
- **list(bucket, prefix)**: Lists all files with given prefix in R2 bucket

### 3. Data Collection Cycle (cycle.py)
Orchestrates the periodic collection of miner data and Bitcoin prices.

#### Global State:
- **miner_data**: Dictionary mapping UID → {uid, hotkey, history, bucket, blocks, btc}

#### Main Function:
- **cycle(netuid, block, mg)**: 
  - Fetches current Bitcoin price from Binance API
  - Retrieves commitment data from Bittensor network
  - Creates mapping from UIDs to hotkeys
  - Generates default zero embedding for absent data
  - Asynchronously processes each UID:
    - Maintains historical data per miner
    - Resets history if hotkey changed (security measure)
    - Downloads latest embedding from miner's R2 bucket
    - Validates payload format and hotkey authenticity
    - Appends new data to history, blocks, and BTC price arrays
  - Updates global miner_data state

### 4. Machine Learning Model (model.py)
Implements salience analysis using a multi-layer perceptron.

#### Neural Network Architecture:
- **MLP Class**: 3-layer neural network with ReLU and Tanh activations
  - **__init__(input_size, hidden_size, output_size)**: Initializes layers
  - **forward(x)**: Forward pass with ReLU → ReLU → Tanh activations

#### Salience Computation:
- **salience(history_dict, btc_prices, hidden_size, lr)**:
  - Constructs feature matrix from all miners' embeddings
  - Trains baseline model to predict Bitcoin price changes
  - For each UID, trains model with that UID's features masked to zero
  - Computes performance degradation (delta loss) when each UID is removed
  - Normalizes delta losses to create weight distribution
  - Returns salience scores indicating each miner's predictive contribution

### 5. Main Validator Loop (main.py)
Coordinates the entire validation process.

#### Utility Functions:
- **commit_r2_bucket(bucket, wallet, hotkey, uid, subtensor)**: Commits bucket to network

#### Core Logic:
- **compute_salience()**:
  - Validates sufficient historical data exists
  - Decrypts timelock-encrypted embeddings from all miners
  - Handles malformed data by substituting zeros
  - Calculates Bitcoin price percentage changes with configured lag
  - Calls salience function to compute predictive importance scores

- **main()**:
  - Parses command-line arguments for wallet and network configuration
  - Initializes Bittensor subtensor and wallet connections
  - Runs continuous monitoring loop:
    - Detects new blocks on the network
    - Updates metagraph and calls cycle() for data collection
    - Every TASK_INTERVAL blocks, spawns background thread to:
      - Compute salience scores
      - Normalize scores to probability distribution
      - Set network weights based on miner performance

### 6. Testing Suite (test_salience.py)
Comprehensive test coverage for all major components.

#### Test Categories:
- **Basic Functionality Tests**:
  - test_salience_all_zeros(): Validates handling of zero inputs
  - test_mlp_initialization(): Verifies neural network setup
  - test_mlp_forward(): Tests forward pass and output constraints

- **Data Variation Tests**:
  - test_salience_with_varying_data(): Tests with non-zero varied inputs
  - test_compute_salience_with_mock_data(): End-to-end test with mocked data

- **Encryption Tests**:
  - test_timelock_roundtrip(): Validates encryption/decryption cycle
  - test_timelock_with_varying_data(): Tests with varied encrypted data
  - test_end_to_end_salience_with_timelock(): Full integration test

## System Workflow

### 1. Initialization Phase
- Validator starts with command-line wallet and network parameters
- Establishes connection to Bittensor subtensor
- Begins monitoring blockchain for new blocks

### 2. Data Collection Phase (Every Block)
- Updates network metagraph to get current miner information
- For each miner UID:
  - Fetches their committed R2 bucket identifier
  - Downloads latest encrypted embedding if available
  - Validates authenticity using hotkey verification
  - Appends to historical data with current Bitcoin price

### 3. Evaluation Phase (Every 360 Blocks)
- Ensures sufficient historical data (> LAG blocks)
- Decrypts all timelock-encrypted embeddings
- Constructs training dataset with lagged Bitcoin returns as targets
- Performs salience analysis to identify most predictive miners
- Converts salience scores to normalized weight distribution

### 4. Weight Setting Phase
- Submits calculated weights to Bittensor network
- Weights determine miner rewards based on their predictive contribution
- Process repeats continuously with new data

## Security Considerations

### Data Integrity
- Timelock encryption prevents future-looking in embeddings
- Hotkey validation ensures authentic data sources
- Historical data reset when miner changes identity

### Fault Tolerance
- Graceful handling of malformed or missing data
- Default zero embeddings for absent miners
- Robust error handling in network and storage operations

## Dependencies

### Core Libraries
- **bittensor**: Blockchain integration and timelock encryption
- **torch**: Neural network implementation and tensor operations
- **aiobotocore/boto3**: Asynchronous cloud storage operations
- **requests**: External API calls for Bitcoin price data

### Supporting Libraries
- **numpy**: Numerical computations
- **pandas**: Data manipulation (if needed)
- **ccxt**: Cryptocurrency exchange integration

## Performance Characteristics

### Computational Complexity
- O(N × T × D) for feature matrix construction (N=miners, T=timesteps, D=dimensions)
- O(N × T × H) for salience computation with H hidden units
- Parallelized data collection reduces latency

### Storage Requirements
- Local caching in ~/storage/ for redundancy
- Cloud storage handles persistent miner data
- Historical data grows linearly with time

## Extensibility

The modular design allows for:
- Different salience algorithms in model.py
- Alternative storage backends in comms.py
- Modified data collection strategies in cycle.py
- Additional evaluation metrics in main.py

This system represents a sophisticated approach to decentralized machine learning evaluation, combining blockchain technology with modern ML techniques for fair and automated miner assessment. 
