#!/usr/bin/env python3
# Initialize price history for immediate feature generation

import asyncio
import time
import numpy as np
from market_data_fallback import FallbackMarketDataFetcher

async def build_initial_price_history():
    """Build initial price history by fetching current price multiple times"""
    print("🔄 Building initial price history...")
    
    prices = []
    async with FallbackMarketDataFetcher() as fetcher:
        # Get current price
        base_price = await fetcher.fetch_current_price()
        if not base_price:
            print("❌ Failed to get base price")
            return []
        
        print(f"📊 Base price: ${base_price:,.2f}")
        
        # Generate synthetic historical prices with realistic variations
        now = time.time()
        for i in range(100, 0, -1):
            # Add small random walk variations (±0.1% per minute)
            variation = np.random.normal(0, 0.001)  # 0.1% standard deviation
            price_change = 1 + variation
            
            # Apply compound changes
            if i < 100:
                price = prices[0] * (price_change ** i)
            else:
                price = base_price * price_change
            
            prices.insert(0, price)
        
        # Add current price as latest
        prices.append(base_price)
        
        print(f"✅ Generated {len(prices)} price points")
        print(f"   Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
        print(f"   Latest: ${prices[-1]:,.2f}")
        
        return prices

async def test_features_with_history():
    """Test feature generation with price history"""
    from market_data_fallback import price_history, fetch_comprehensive_market_data
    
    # Build initial history
    initial_prices = await build_initial_price_history()
    
    # Update global price history
    price_history.clear()
    price_history.extend(initial_prices)
    
    print(f"\n📈 Price history updated: {len(price_history)} points")
    
    # Test feature generation
    features = await fetch_comprehensive_market_data()
    
    if features is not None:
        print(f"✅ Features generated: {features.shape}")
        print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   Non-zero: {np.count_nonzero(features)}/{len(features)}")
        print(f"   Sample features: {features[:10]}")
        return True
    else:
        print("❌ Feature generation failed")
        return False

if __name__ == "__main__":
    import numpy as np
    result = asyncio.run(test_features_with_history())
    print(f"\n🎯 Test result: {'✅ SUCCESS' if result else '❌ FAILED'}")