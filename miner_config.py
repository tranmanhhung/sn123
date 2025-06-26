# MIT License - MANTIS Miner Configuration

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class MinerConfig:
    # Miner Identity
    hotkey: str = "5HMci1Z3VS5QkdvDiWc8vLiwwD7YJb7VfTKEfmBwiD6G7joh"  # Replace with your hotkey
    wallet_name: str = "c3"  # Replace with your wallet name
    hotkey_name: str = "h1"   # Replace with your hotkey name
    
    # Network Settings
    netuid: int = 123
    network: str = "finney"
    
    # Timelock Settings
    lock_time_seconds: int = 30
    feature_length: int = 100
    
    # Drand Configuration
    drand_api: str = "https://api.drand.sh/v2"
    drand_beacon_id: str = "quicknet"
    drand_public_key: str = (
        "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
        "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
        "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
    )
    
    # Mining Loop Settings
    mining_interval: int = 60  # seconds between predictions
    retry_delay: int = 30      # seconds to wait after errors
    max_retries: int = 3
    
    # R2 Storage (from environment variables)
    r2_account_id: Optional[str] = None
    r2_access_key_id: Optional[str] = None
    r2_secret_access_key: Optional[str] = None
    
    # Market Data Settings - Multiple endpoints for geo-restrictions
    binance_api_base: str = "https://api.binance.com/api/v3"
    binance_api_alternatives: list = None
    coingecko_api_base: str = "https://api.coingecko.com/api/v3"
    
    # ML Model Settings
    lookback_periods: int = 100   # number of historical data points to use
    prediction_horizon: int = 10  # minutes ahead to predict (matches validator LAG)
    
    def __post_init__(self):
        # Load R2 credentials from environment
        self.r2_account_id = os.getenv("R2_ACCOUNT_ID")
        self.r2_access_key_id = os.getenv("R2_WRITE_ACCESS_KEY_ID") 
        self.r2_secret_access_key = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
        
        # Setup alternative Binance endpoints for geo-restrictions
        self.binance_api_alternatives = [
            "https://api.binance.com/api/v3",
            "https://api1.binance.com/api/v3", 
            "https://api2.binance.com/api/v3",
            "https://api3.binance.com/api/v3",
            "https://api.binance.us/api/v3",  # US endpoint
            "https://fapi.binance.com/fapi/v1",  # Futures API (sometimes less restricted)
        ]
        
        if not all([self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key]):
            raise ValueError("Missing R2 credentials in environment variables")
    
    @property
    def r2_endpoint_url(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
    
    @property 
    def r2_public_url(self) -> str:
        # Correct public URL format (direct object access)
        return f"https://pub-09f5de84cf104b3f8ce32bb6b2d774f9.r2.dev/{self.hotkey}"
    
    @property
    def r2_bucket_name(self) -> str:
        # Use existing bucket name instead of hotkey
        return "miner-c8-h1"

# Global config instance
config = MinerConfig()