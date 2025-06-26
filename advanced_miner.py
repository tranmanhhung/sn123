#!/usr/bin/env python3
# MIT License - MANTIS Advanced Miner

"""
Advanced MANTIS Miner with:
- Automated prediction loop
- ML-based Bitcoin price prediction  
- Robust error handling and retry logic
- Performance monitoring and metrics
- Adaptive strategies
"""

import asyncio
import json
import os
import secrets
import signal
import sys
import time
from typing import Optional
import argparse
import traceback

import bittensor as bt
import boto3
import numpy as np
import requests
from dotenv import load_dotenv
from timelock import Timelock

# Import our custom modules
from miner_config import config
from market_data_fallback import fetch_comprehensive_market_data
from ml_predictor import predictor
from miner_monitoring import monitor

class AdvancedMiner:
    """Advanced miner with automation and AI prediction"""
    
    def __init__(self):
        self.running = False
        self.wallet: Optional[bt.wallet] = None
        self.subtensor: Optional[bt.subtensor] = None
        self.s3_client = None
        self.timelock = Timelock(config.drand_public_key)
        
        # Performance tracking
        self.consecutive_errors = 0
        self.last_successful_cycle = 0
        self.adaptive_interval = config.mining_interval
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        monitor.logger.info("Advanced Miner initialized")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        monitor.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def setup_bittensor(self, wallet_name: str, hotkey_name: str):
        """Initialize Bittensor wallet and subtensor"""
        try:
            self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            self.subtensor = bt.subtensor(network=config.network)
            
            # Verify wallet is registered
            if not self.subtensor.is_hotkey_registered(
                netuid=config.netuid, 
                hotkey_ss58=self.wallet.hotkey.ss58_address
            ):
                raise ValueError(f"Hotkey {self.wallet.hotkey.ss58_address} not registered on subnet {config.netuid}")
            
            monitor.logger.info(f"Bittensor setup complete for {self.wallet.hotkey.ss58_address}")
            return True
            
        except Exception as e:
            monitor.logger.error(f"Failed to setup Bittensor: {e}")
            return False
    
    def setup_r2_client(self):
        """Initialize R2 S3 client"""
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=config.r2_endpoint_url,
                aws_access_key_id=config.r2_access_key_id,
                aws_secret_access_key=config.r2_secret_access_key,
                region_name="auto"
            )
            
            # Test connection by listing bucket
            self.s3_client.head_bucket(Bucket=config.r2_bucket_name)
            monitor.logger.info("R2 client setup successful")
            return True
            
        except Exception as e:
            monitor.logger.error(f"Failed to setup R2 client: {e}")
            return False
    
    async def commit_url_to_network(self) -> bool:
        """Commit miner URL to Bittensor network (one-time setup)"""
        try:
            if not self.subtensor or not self.wallet:
                return False
            
            # Check if already committed
            metagraph = bt.metagraph(netuid=config.netuid, network=config.network, sync=True)
            commitments = self.subtensor.get_all_commitments(config.netuid)
            
            if self.wallet.hotkey.ss58_address in commitments:
                current_url = commitments[self.wallet.hotkey.ss58_address]
                if current_url == config.r2_public_url:
                    monitor.logger.info("URL already committed correctly")
                    return True
                else:
                    monitor.logger.info(f"Updating committed URL from {current_url} to {config.r2_public_url}")
            
            # Commit URL
            result = self.subtensor.commit(
                wallet=self.wallet,
                netuid=config.netuid,
                data=config.r2_public_url
            )
            
            if result:
                monitor.logger.info(f"Successfully committed URL: {config.r2_public_url}")
                return True
            else:
                monitor.logger.error("Failed to commit URL to network")
                return False
                
        except Exception as e:
            monitor.logger.error(f"Error committing URL: {e}")
            return False
    
    async def get_drand_round_for_future(self, seconds_ahead: int = 30) -> Optional[int]:
        """Get Drand round number for future time"""
        try:
            response = requests.get(
                f"{config.drand_api}/beacons/{config.drand_beacon_id}/info",
                timeout=10
            )
            response.raise_for_status()
            info = response.json()
            
            future_time = time.time() + seconds_ahead
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            monitor.logger.debug(f"Target Drand round: {target_round}")
            return target_round
            
        except Exception as e:
            monitor.logger.error(f"Failed to get Drand round: {e}")
            monitor.record_network_error(str(e))
            return None
    
    async def generate_prediction(self) -> Optional[np.ndarray]:
        """Generate ML-based Bitcoin price prediction"""
        try:
            monitor.logger.info("Generating market prediction...")
            
            # Fetch comprehensive market data
            market_features = await fetch_comprehensive_market_data()
            if market_features is None:
                monitor.logger.error("Failed to fetch market data")
                return None
            
            # Get current price for training data using fallback
            current_price = None
            try:
                from market_data_fallback import FallbackMarketDataFetcher
                async with FallbackMarketDataFetcher() as fetcher:
                    current_price = await fetcher.fetch_current_price()
                
                if current_price is None:
                    current_price = 50000.0  # Fallback
                    monitor.logger.warning("Using fallback price")
            except Exception as e:
                monitor.logger.warning(f"Failed to get current price: {e}")
                current_price = 50000.0  # Fallback
            
            # Add training data to predictor (for future price target, we'll update later)
            predictor.add_training_data(market_features, current_price)
            
            # Retrain model if needed
            predictor.retrain_if_needed()
            
            # Generate prediction
            prediction_vector = predictor.predict(market_features, current_price)
            
            # Extract prediction strength for monitoring
            # (This is a rough estimate since we encoded it in the vector)
            prediction_strength = np.mean(prediction_vector[-10:])  # Last 10 features contain our encoding
            confidence = min(1.0, abs(prediction_strength))
            
            monitor.record_prediction(prediction_strength, confidence)
            
            monitor.logger.info(f"Prediction generated with strength: {prediction_strength:.4f}")
            return prediction_vector
            
        except Exception as e:
            monitor.logger.error(f"Error generating prediction: {e}")
            monitor.record_network_error(str(e))
            return None
    
    async def encrypt_prediction(self, prediction: np.ndarray) -> Optional[dict]:
        """Encrypt prediction using timelock"""
        try:
            # Get target Drand round
            target_round = await self.get_drand_round_for_future(config.lock_time_seconds)
            if target_round is None:
                return None
            
            # Convert prediction to string format
            prediction_list = prediction.tolist()
            prediction_str = str(prediction_list)
            
            # Generate random salt
            salt = secrets.token_bytes(32)
            
            # Encrypt using timelock
            ciphertext = self.timelock.tle(target_round, prediction_str, salt)
            ciphertext_hex = ciphertext.hex()
            
            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            monitor.logger.info(f"Prediction encrypted for round {target_round}")
            return payload
            
        except Exception as e:
            monitor.logger.error(f"Error encrypting prediction: {e}")
            return None
    
    async def upload_to_r2(self, payload: dict) -> bool:
        """Upload encrypted payload to R2 storage"""
        try:
            if not self.s3_client:
                return False
            
            # Convert payload to JSON
            payload_json = json.dumps(payload, indent=2)
            
            # Upload to R2
            self.s3_client.put_object(
                Bucket=config.r2_bucket_name,  # Use bucket from config
                Key=config.hotkey,              # Object key = hotkey
                Body=payload_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            monitor.record_upload_success()
            monitor.logger.info("Payload uploaded successfully to R2")
            return True
            
        except Exception as e:
            monitor.record_upload_failure(str(e))
            monitor.logger.error(f"Failed to upload to R2: {e}")
            return False
    
    async def mining_cycle(self) -> bool:
        """Execute one complete mining cycle"""
        cycle_start = time.time()
        
        try:
            monitor.logger.info("Starting mining cycle...")
            
            # Generate prediction
            prediction = await self.generate_prediction()
            if prediction is None:
                return False
            
            # Encrypt prediction
            encrypted_payload = await self.encrypt_prediction(prediction)
            if encrypted_payload is None:
                return False
            
            # Upload to R2
            upload_success = await self.upload_to_r2(encrypted_payload)
            if not upload_success:
                return False
            
            # Update success metrics
            self.consecutive_errors = 0
            self.last_successful_cycle = time.time()
            
            cycle_duration = time.time() - cycle_start
            monitor.logger.info(f"Mining cycle completed in {cycle_duration:.2f}s")
            
            return True
            
        except Exception as e:
            monitor.logger.error(f"Mining cycle failed: {e}")
            monitor.logger.debug(traceback.format_exc())
            return False
    
    def adaptive_sleep(self, success: bool):
        """Adaptive sleep based on recent performance"""
        if success:
            # Successful cycle - use normal interval
            if self.adaptive_interval > config.mining_interval:
                # Gradually return to normal interval
                self.adaptive_interval = max(
                    config.mining_interval,
                    self.adaptive_interval * 0.9
                )
        else:
            # Failed cycle - increase interval (backoff)
            self.consecutive_errors += 1
            self.adaptive_interval = min(
                config.mining_interval * 3,  # Max 3x normal interval
                config.mining_interval * (1.5 ** self.consecutive_errors)
            )
        
        monitor.logger.info(f"Sleeping for {self.adaptive_interval:.1f}s...")
        time.sleep(self.adaptive_interval)
    
    async def run_mining_loop(self):
        """Main mining loop with error handling and monitoring"""
        monitor.logger.info("Starting mining loop...")
        self.running = True
        
        # Status reporting interval
        last_status_report = 0
        status_interval = 1800  # 30 minutes
        
        while self.running:
            try:
                # Execute mining cycle
                success = await self.mining_cycle()
                
                # Adaptive sleep based on performance
                self.adaptive_sleep(success)
                
                # Periodic status reporting
                current_time = time.time()
                if current_time - last_status_report > status_interval:
                    monitor.log_status_summary()
                    health = monitor.check_health()
                    if not all(health.values()):
                        monitor.logger.warning(f"Health check issues: {health}")
                    last_status_report = current_time
                
            except Exception as e:
                monitor.logger.error(f"Unexpected error in mining loop: {e}")
                monitor.logger.debug(traceback.format_exc())
                monitor.record_network_error(str(e))
                
                # Longer sleep on unexpected errors
                time.sleep(config.retry_delay)
        
        monitor.logger.info("Mining loop stopped")
    
    async def initialize(self, wallet_name: str, hotkey_name: str) -> bool:
        """Initialize all miner components"""
        monitor.logger.info("Initializing miner...")
        
        # Setup Bittensor
        if not self.setup_bittensor(wallet_name, hotkey_name):
            return False
        
        # Setup R2 client  
        if not self.setup_r2_client():
            return False
        
        # Commit URL to network (one-time setup)
        if not await self.commit_url_to_network():
            monitor.logger.warning("Failed to commit URL, but continuing...")
        
        monitor.logger.info("Miner initialization complete")
        return True

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MANTIS Advanced Miner")
    parser.add_argument("--wallet.name", required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", required=True, help="Hotkey name")
    parser.add_argument("--commit-only", action="store_true", 
                       help="Only commit URL and exit")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Update config with provided values
    config.wallet_name = getattr(args, 'wallet.name')
    config.hotkey_name = getattr(args, 'wallet.hotkey')
    
    # Create miner instance
    miner = AdvancedMiner()
    
    try:
        # Initialize miner
        if not await miner.initialize(config.wallet_name, config.hotkey_name):
            monitor.logger.error("Failed to initialize miner")
            return 1
        
        if getattr(args, 'commit_only', False):
            monitor.logger.info("Commit-only mode, exiting...")
            return 0
        
        # Start mining loop
        await miner.run_mining_loop()
        
    except KeyboardInterrupt:
        monitor.logger.info("Received interrupt signal")
    except Exception as e:
        monitor.logger.error(f"Fatal error: {e}")
        monitor.logger.debug(traceback.format_exc())
        return 1
    finally:
        # Save final metrics
        monitor.save_metrics()
        monitor.logger.info("Miner shutdown complete")
    
    return 0

if __name__ == "__main__":
    # Install required packages check
    try:
        import pandas
        import sklearn
        import aiohttp
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install: pip install pandas scikit-learn aiohttp")
        sys.exit(1)
    
    # Run main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)