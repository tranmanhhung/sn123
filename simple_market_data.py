# Simple market data fetcher for geo-restricted regions

import asyncio
import logging
import time
import numpy as np
from typing import Optional
import aiohttp
from miner_config import config

logger = logging.getLogger(__name__)

# Global price history
price_history = []

async def get_bitcoin_price_simple() -> Optional[float]:
    """Get Bitcoin price from free APIs that usually work globally"""
    
    # Try multiple free APIs
    apis = [
        {
            'url': 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
            'extract': lambda data: float(data['bitcoin']['usd'])
        },
        {
            'url': 'https://api.coinbase.com/v2/exchange-rates?currency=BTC', 
            'extract': lambda data: float(data['data']['rates']['USD'])
        },
        {
            'url': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
            'extract': lambda data: float(list(data['result'].values())[0]['c'][0])
        },
        {
            'url': 'https://api.gemini.com/v1/pubticker/btcusd',
            'extract': lambda data: float(data['last'])
        },
        {
            'url': 'https://api.bitfinex.com/v1/pubticker/btcusd',
            'extract': lambda data: float(data['last_price'])
        }
    ]
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for api in apis:
            try:
                logger.debug(f"Trying API: {api['url']}")
                async with session.get(api['url'], headers={'User-Agent': 'MANTIS-Miner/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = api['extract'](data)
                        
                        # Sanity check
                        if 10000 < price < 200000:
                            logger.info(f"Bitcoin price: ${price:,.2f}")
                            return price
                        
            except Exception as e:
                logger.debug(f"API failed: {e}")
                continue
    
    logger.error("All Bitcoin price APIs failed")
    return None

async def fetch_comprehensive_market_data() -> Optional[np.ndarray]:
    """Simplified market data for geo-restricted regions"""
    global price_history
    
    try:
        # Get current price
        current_price = await get_bitcoin_price_simple()
        
        if current_price is None:
            logger.error("Cannot fetch Bitcoin price")
            return None
        
        # Update price history
        price_history.append(current_price)
        
        # Keep only last 100 prices
        if len(price_history) > 100:
            price_history = price_history[-100:]
        
        # Generate features based on price history
        features = generate_simple_features(price_history)
        
        logger.info(f"Generated {len(features)} features from price history")
        return features
        
    except Exception as e:
        logger.error(f"Error in market data fetch: {e}")
        return None

def generate_simple_features(prices: list) -> np.ndarray:
    """Generate features from price history only"""
    
    if len(prices) < 10:
        # Not enough history, return random features
        logger.warning("Insufficient price history, using random features")
        return np.random.uniform(-0.1, 0.1, config.feature_length).astype(np.float32)
    
    prices = np.array(prices)
    features = []
    
    # Price-based features
    current_price = prices[-1]
    
    # Returns over different periods
    for period in [1, 2, 3, 5, 10, 20]:
        if len(prices) > period:
            ret = (current_price - prices[-period-1]) / prices[-period-1]
            features.append(ret)
    
    # Moving averages ratios
    for window in [3, 5, 10, 20]:
        if len(prices) >= window:
            ma = np.mean(prices[-window:])
            features.append((current_price - ma) / ma)
    
    # Volatility measures
    for window in [5, 10, 20]:
        if len(prices) > window:
            returns = np.diff(prices[-window:]) / prices[-window:-1]
            vol = np.std(returns)
            features.append(vol)
    
    # Price percentiles
    if len(prices) >= 20:
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            pct = np.percentile(prices[-20:], p)
            features.append((current_price - pct) / pct)
    
    # Momentum indicators
    for lookback in [3, 5, 10]:
        if len(prices) > lookback:
            momentum = (current_price - prices[-lookback]) / prices[-lookback]
            features.append(momentum)
    
    # Trend indicators (linear regression slope)
    for window in [5, 10, 20]:
        if len(prices) >= window:
            x = np.arange(window)
            y = prices[-window:]
            slope = np.polyfit(x, y, 1)[0]
            features.append(slope / current_price)  # Normalize by price
    
    # High-low ratios
    for window in [5, 10, 20]:
        if len(prices) >= window:
            window_prices = prices[-window:]
            high_low_ratio = (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices)
            features.append(high_low_ratio)
    
    # Price position in recent range
    for window in [10, 20]:
        if len(prices) >= window:
            window_prices = prices[-window:]
            min_price = np.min(window_prices)
            max_price = np.max(window_prices)
            if max_price > min_price:
                position = (current_price - min_price) / (max_price - min_price)
                features.append(position)
    
    # Fill remaining with variations of existing features
    while len(features) < config.feature_length:
        if len(features) > 0:
            # Add scaled versions of existing features
            idx = len(features) % min(len(features), 50)
            if idx < len(features):
                scaled_feature = features[idx] * np.random.uniform(0.5, 1.5)
                features.append(scaled_feature)
            else:
                features.append(np.random.uniform(-0.01, 0.01))
        else:
            features.append(0.0)
    
    # Ensure exactly the right length
    features = np.array(features[:config.feature_length], dtype=np.float32)
    
    # Normalize to [-1, 1] range
    features = np.clip(features, -10, 10)  # Clip extreme values
    features = np.tanh(features)  # Smooth normalization
    
    return features

# Test function
async def test_simple_market_data():
    """Test the simple market data fetcher"""
    print("Testing simple market data fetcher...")
    
    price = await get_bitcoin_price_simple()
    if price:
        print(f"✅ Bitcoin price: ${price:,.2f}")
        
        # Generate features
        features = await fetch_comprehensive_market_data()
        if features is not None:
            print(f"✅ Generated {len(features)} features")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
            return True
    
    print("❌ Failed to fetch market data")
    return False

if __name__ == "__main__":
    asyncio.run(test_simple_market_data())