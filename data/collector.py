import ccxt
import pandas as pd
from datetime import datetime

class DataCollector:
    """
    암호화폐 거래소에서 과거 가격 데이터 수집
    
    기본: Binance, BTC/USDT, 1시간봉
    """
    
    def __init__(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        초기화
        
        Args:
            exchange: 거래소 이름 (binance, coinbase 등)
            symbol: 거래 쌍 (BTC/USDT, ETH/USDT 등)
            timeframe: 시간 단위 (1m, 5m, 1h, 1d 등)
        """
        self.exchange_name = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        
        # 거래소 객체 생성
        try:
            self.exchange = getattr(ccxt, exchange)()
            print(f"✅ {exchange} 거래소 연결 성공")
        except Exception as e:
            print(f"❌ 거래소 연결 실패: {e}")
            raise
    
    def fetch_ohlcv(self, limit=1000):
        """
        OHLCV 데이터 수집
        
        Args:
            limit: 가져올 캔들 개수 (최대 1000)
        
        Returns:
            pandas DataFrame [timestamp, open, high, low, close, volume]
        """
        print(f"\n📊 데이터 수집 중...")
        print(f"   거래소: {self.exchange_name}")
        print(f"   심볼: {self.symbol}")
        print(f"   시간봉: {self.timeframe}")
        print(f"   개수: {limit}개")
        
        try:
            # API 호출
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            # DataFrame 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환 (밀리초 → 날짜)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"✅ {len(df)}개 데이터 수집 완료\n")
            
            # 기본 정보 출력
            print("=" * 50)
            print("📈 데이터 미리보기")
            print("=" * 50)
            print(df.head())
            print("\n" + "=" * 50)
            print("📊 기본 통계")
            print("=" * 50)
            print(df.describe())
            print("\n" + "=" * 50)
            print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"일수: {(df['timestamp'].max() - df['timestamp'].min()).days}일")
            print("=" * 50)
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            raise
    
    def get_latest_price(self):
        """
        현재 가격 조회
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ 가격 조회 실패: {e}")
            return None
    
    def save_to_csv(self, df, filename='data.csv'):
        """
        데이터를 CSV 파일로 저장
        """
        df.to_csv(filename, index=False)
        print(f"💾 데이터 저장: {filename}")


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 암호화폐 데이터 수집기 V0.1")
    print("=" * 50)
    
    # 1. 수집기 생성
    collector = DataCollector(
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h'
    )
    
    # 2. 현재 가격 확인
    current_price = collector.get_latest_price()
    if current_price:
        print(f"\n💰 현재 비트코인 가격: ${current_price:,.2f}\n")
    
    # 3. 과거 데이터 수집
    df = collector.fetch_ohlcv(limit=1000)
    
    # 4. CSV 저장
    collector.save_to_csv(df, 'btc_1h_data.csv')
    
    print("\n✅ 모든 작업 완료!")