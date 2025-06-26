# MIT License - MANTIS Market Data Fetcher

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from miner_config import config

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    timestamp: float
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float

class MarketDataFetcher:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_history: List[MarketDataPoint] = []
        self.last_fetch_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_current_price(self) -> Optional[float]:
        """Fetch current Bitcoin price from Binance"""
        try:
            url = f"{config.binance_api_base}/ticker/price"
            params = {"symbol": "BTCUSDT"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                else:
                    logger.error(f"Binance API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None
    
    async def fetch_klines(self, symbol: str = "BTCUSDT", 
                          interval: str = "1m", 
                          limit: int = 100) -> List[MarketDataPoint]:
        """Fetch historical kline data from Binance"""
        try:
            url = f"{config.binance_api_base}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    market_data = []
                    for kline in data:
                        point = MarketDataPoint(
                            timestamp=float(kline[0]) / 1000,  # Convert to seconds
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                            price=float(kline[4]),  # Use close as price
                            volume=float(kline[5])
                        )
                        market_data.append(point)
                    
                    return market_data
                else:
                    logger.error(f"Binance klines API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return []
    
    async def fetch_orderbook(self, symbol: str = "BTCUSDT", 
                             limit: int = 100) -> Optional[Dict]:
        """Fetch order book data for market depth analysis"""
        try:
            url = f"{config.binance_api_base}/depth"
            params = {
                "symbol": symbol,
                "limit": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Orderbook API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None
    
    async def fetch_24hr_ticker(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Fetch 24hr ticker statistics"""
        try:
            url = f"{config.binance_api_base}/ticker/24hr"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"24hr ticker API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching 24hr ticker: {e}")
            return None

class FeatureEngineer:
    """Extract meaningful features from market data for ML model"""
    
    @staticmethod
    def calculate_technical_indicators(data: List[MarketDataPoint]) -> np.ndarray:
        """Calculate technical indicators from price data"""
        if len(data) < 20:
            logger.warning("Insufficient data for technical indicators")
            return np.zeros(20)  # Return zeros if not enough data
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame([{
            'price': point.price,
            'volume': point.volume,
            'high': point.high,
            'low': point.low,
            'open': point.open,
            'close': point.close
        } for point in data])
        
        features = []
        
        # Price-based features
        features.append(df['close'].iloc[-1])  # Current price
        features.append(df['close'].pct_change().iloc[-1])  # Price change %
        features.append(df['close'].pct_change(5).iloc[-1])  # 5-period change %
        features.append(df['close'].pct_change(10).iloc[-1])  # 10-period change %
        
        # Moving averages
        features.append(df['close'].rolling(5).mean().iloc[-1])   # SMA 5
        features.append(df['close'].rolling(10).mean().iloc[-1])  # SMA 10
        features.append(df['close'].rolling(20).mean().iloc[-1])  # SMA 20
        
        # Volatility
        features.append(df['close'].rolling(10).std().iloc[-1])  # 10-period volatility
        features.append(df['close'].rolling(20).std().iloc[-1])  # 20-period volatility
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1])
        
        # Volume features
        features.append(df['volume'].iloc[-1])  # Current volume
        features.append(df['volume'].rolling(5).mean().iloc[-1])   # Volume SMA 5
        features.append(df['volume'].rolling(10).mean().iloc[-1])  # Volume SMA 10
        
        # Price-Volume relationship
        features.append(df['close'].iloc[-1] * df['volume'].iloc[-1])  # Price * Volume
        
        # High-Low spread
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_position = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        features.append(bb_position)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        features.append(macd.iloc[-1])
        
        # Momentum indicators
        features.append(df['close'].iloc[-1] - df['close'].iloc[-5])   # 5-period momentum
        features.append(df['close'].iloc[-1] - df['close'].iloc[-10])  # 10-period momentum
        
        # Rate of Change
        features.append((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10])
        
        # Pad with zeros if we have NaN values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    @staticmethod
    def calculate_market_depth_features(orderbook: Dict) -> np.ndarray:
        """Calculate features from order book data"""
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return np.zeros(10)
        
        features = []
        
        # Bid-Ask spread
        best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
        best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
        features.append(spread)
        
        # Order book imbalance
        bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:10])
        ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:10])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        features.append(imbalance)
        
        # Weighted mid price
        if bid_volume + ask_volume > 0:
            weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        else:
            weighted_mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
        features.append(weighted_mid)
        
        # Add more depth features
        for i in range(7):
            features.append(0.0)  # Placeholder for additional depth features
        
        return np.array(features, dtype=np.float32)

async def fetch_comprehensive_market_data() -> Optional[np.ndarray]:
    """Fetch and process comprehensive market data for ML model"""
    try:
        async with MarketDataFetcher() as fetcher:
            # Fetch various data sources
            klines = await fetcher.fetch_klines(limit=config.lookback_periods)
            orderbook = await fetcher.fetch_orderbook()
            ticker_24hr = await fetcher.fetch_24hr_ticker()
            
            if not klines:
                logger.error("Failed to fetch klines data")
                return None
            
            # Calculate technical indicators
            tech_features = FeatureEngineer.calculate_technical_indicators(klines)
            
            # Calculate market depth features
            depth_features = FeatureEngineer.calculate_market_depth_features(orderbook)
            
            # 24hr ticker features
            ticker_features = []
            if ticker_24hr:
                ticker_features.extend([
                    float(ticker_24hr.get('priceChangePercent', 0)),
                    float(ticker_24hr.get('weightedAvgPrice', 0)),
                    float(ticker_24hr.get('volume', 0)),
                    float(ticker_24hr.get('count', 0))  # Trade count
                ])
            else:
                ticker_features = [0.0] * 4
            
            # Combine all features
            all_features = np.concatenate([
                tech_features,
                depth_features, 
                np.array(ticker_features, dtype=np.float32)
            ])
            
            # Normalize features to [-1, 1] range
            # Note: In production, you'd want to use pre-computed normalization parameters
            normalized_features = np.tanh(all_features / np.std(all_features + 1e-8))
            
            # Ensure we have exactly 100 features as required
            if len(normalized_features) > config.feature_length:
                normalized_features = normalized_features[:config.feature_length]
            elif len(normalized_features) < config.feature_length:
                # Pad with zeros
                padding = np.zeros(config.feature_length - len(normalized_features))
                normalized_features = np.concatenate([normalized_features, padding])
            
            logger.info(f"Generated {len(normalized_features)} market features")
            return normalized_features
            
    except Exception as e:
        logger.error(f"Error fetching comprehensive market data: {e}")
        return None