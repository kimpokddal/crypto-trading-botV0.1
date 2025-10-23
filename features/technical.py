import pandas as pd
import numpy as np

class TechnicalFeatures:
    """
    기술적 지표 계산 및 피처 생성
    
    이동평균 기반 피처:
    - ma5, ma20, ma50 (단기/중기/장기 추세)
    - ma5_20_diff (단기 모멘텀)
    - ma20_50_diff (중기 모멘텀)
    """
    
    def __init__(self, df):
        """
        초기화
        
        Args:
            df: OHLCV 데이터프레임
        """
        self.df = df.copy()  # 원본 보존
        print(f"📊 입력 데이터: {len(self.df)}개 캔들")
    
    def add_moving_averages(self, periods=[5, 20, 50]):
        """
        이동평균선 추가
        
        Args:
            periods: 이동평균 기간 리스트
        
        Returns:
            DataFrame with moving averages
        """
        print(f"\n🔄 이동평균 계산 중...")
        
        for period in periods:
            col_name = f'ma{period}'
            self.df[col_name] = self.df['close'].rolling(
                window=period
            ).mean()
            print(f"   ✅ {col_name} 계산 완료")
        
        # NaN 제거 (초기 데이터 부족)
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        
        print(f"\n   ⚠️  NaN 제거: {before} → {after}개 ({before-after}개 제거)")
        
        return self.df
    
    def add_momentum_features(self):
        """
        모멘텀 피처 추가
        
        모멘텀 = 이동평균 간 차이
        - 단기 - 중기 = 단기 모멘텀
        - 중기 - 장기 = 중기 모멘텀
        """
        print(f"\n⚡ 모멘텀 피처 생성 중...")
        
        # 단기 모멘텀 (ma5 - ma20)
        self.df['ma5_20_diff'] = self.df['ma5'] - self.df['ma20']
        print(f"   ✅ ma5_20_diff (단기 모멘텀)")
        
        # 중기 모멘텀 (ma20 - ma50)
        self.df['ma20_50_diff'] = self.df['ma20'] - self.df['ma50']
        print(f"   ✅ ma20_50_diff (중기 모멘텀)")
        
        return self.df
    
    def add_labels(self):
        """
        레이블 추가: 미래 수익률
        
        future_return = (다음 가격 - 현재 가격) / 현재 가격
        """
        print(f"\n🎯 레이블 생성 중...")
        
        # 수익률 계산
        self.df['future_return'] = self.df['close'].pct_change().shift(-1)
        
        # 마지막 행은 미래 데이터 없음 → NaN 제거
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        
        print(f"   ✅ future_return 계산 완료")
        print(f"   ⚠️  미래 데이터 없는 행 제거: {before} → {after}개")
        
        return self.df
    
    def get_features_and_labels(self):
        """
        학습용 피처와 레이블 분리
        
        Returns:
            X (features), y (labels)
        """
        feature_cols = ['ma5', 'ma20', 'ma50', 'ma5_20_diff', 'ma20_50_diff']
        
        X = self.df[feature_cols].values
        y = self.df['future_return'].values
        
        print(f"\n✅ 학습 데이터 준비 완료")
        print(f"   피처 (X): {X.shape}")
        print(f"   레이블 (y): {y.shape}")
        
        return X, y
    
    def show_statistics(self):
        """
        기본 통계 출력
        """
        print(f"\n" + "=" * 60)
        print(f"📈 피처 통계")
        print(f"=" * 60)
        
        feature_cols = ['ma5', 'ma20', 'ma50', 'ma5_20_diff', 'ma20_50_diff']
        print(self.df[feature_cols].describe())
        
        print(f"\n" + "=" * 60)
        print(f"🎯 레이블 통계 (future_return)")
        print(f"=" * 60)
        print(f"평균: {self.df['future_return'].mean():.6f}")
        print(f"표준편차: {self.df['future_return'].std():.6f}")
        print(f"최소: {self.df['future_return'].min():.6f}")
        print(f"최대: {self.df['future_return'].max():.6f}")
        
        # 방향성 분석
        positive = (self.df['future_return'] > 0).sum()
        negative = (self.df['future_return'] < 0).sum()
        total = len(self.df)
        
        print(f"\n방향성:")
        print(f"   상승 (양수): {positive}개 ({positive/total*100:.1f}%)")
        print(f"   하락 (음수): {negative}개 ({negative/total*100:.1f}%)")
        print(f"=" * 60)


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 기술적 피처 생성기 V0.1")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n📂 데이터 로딩 중...")
    df = pd.read_csv('btc_1h_data.csv')
    print(f"✅ {len(df)}개 캔들 로드 완료")
    
    # 2. 피처 생성기 초기화
    tech = TechnicalFeatures(df)
    
    # 3. 이동평균 계산
    tech.add_moving_averages(periods=[5, 20, 50])
    
    # 4. 모멘텀 피처 추가
    tech.add_momentum_features()
    
    # 5. 레이블 추가
    tech.add_labels()
    
    # 6. 통계 확인
    tech.show_statistics()
    
    # 7. 학습용 데이터 추출
    X, y = tech.get_features_and_labels()
    
    # 8. 샘플 확인
    print(f"\n" + "=" * 60)
    print(f"📋 샘플 데이터 (처음 5개)")
    print(f"=" * 60)
    print(f"\n피처 (X):")
    print(X[:5])
    print(f"\n레이블 (y):")
    print(y[:5])
    print(f"=" * 60)
    
    # 9. 전처리된 데이터 저장
    tech.df.to_csv('btc_features.csv', index=False)
    print(f"\n💾 전처리 데이터 저장: btc_features.csv")
    
    print(f"\n✅ 모든 작업 완료!")