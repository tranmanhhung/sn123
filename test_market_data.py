#!/usr/bin/env python3
# Test market data fetching with geo-restriction fallbacks

import asyncio
import sys
import traceback

async def test_market_data():
    """Test market data fetching with multiple endpoints"""
    print("=== Testing Market Data Fallbacks ===")
    
    try:
        from market_data_fallback import FallbackMarketDataFetcher, fetch_comprehensive_market_data
        
        # Test individual endpoints
        async with FallbackMarketDataFetcher() as fetcher:
            print("\n🔍 Testing individual endpoints:")
            
            for i, source in enumerate(fetcher.data_sources):
                print(f"\n{i+1}. Testing {source['name']}...")
                is_working = await fetcher.test_endpoint(source)
                status = "✅ Working" if is_working else "❌ Failed/Blocked"
                print(f"   Status: {status}")
                
                if is_working:
                    try:
                        price = await fetcher.fetch_current_price()
                        if price:
                            print(f"   Price: ${price:,.2f}")
                        break
                    except Exception as e:
                        print(f"   Error getting price: {e}")
            
            print(f"\n🎯 Working endpoint: {fetcher.working_endpoint['name'] if fetcher.working_endpoint else 'None'}")
        
        # Test comprehensive data fetch
        print("\n📊 Testing comprehensive market data...")
        market_features = await fetch_comprehensive_market_data()
        
        if market_features is not None:
            print(f"✅ Market features generated: {market_features.shape}")
            print(f"   Feature range: [{market_features.min():.3f}, {market_features.max():.3f}]")
            print(f"   Non-zero features: {np.count_nonzero(market_features)}")
        else:
            print("❌ Failed to generate market features")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

def test_simple_requests():
    """Test simple requests to different endpoints"""
    import requests
    
    print("\n=== Testing Simple HTTP Requests ===")
    
    endpoints = [
        ("Binance Main", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"),
        ("Binance US", "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"), 
        ("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"),
        ("Coinbase", "https://api.coinbase.com/v2/exchange-rates?currency=BTC"),
        ("Kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD")
    ]
    
    for name, url in endpoints:
        try:
            print(f"\n🌐 Testing {name}...")
            response = requests.get(url, timeout=10, headers={'User-Agent': 'MANTIS-Test/1.0'})
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success - Response size: {len(str(data))} chars")
            elif response.status_code == 451:
                print(f"   ❌ Geo-blocked (451)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Test simple requests first
    test_simple_requests()
    
    # Test async market data
    try:
        import numpy as np
        asyncio.run(test_market_data())
    except Exception as e:
        print(f"Async test failed: {e}")
    
    print("\n=== Test Complete ===")