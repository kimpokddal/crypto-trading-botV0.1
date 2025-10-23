import torch
import numpy as np
import pandas as pd
import sys
import os

# === 경로 설정 ===
CURRENT_FILE = os.path.abspath(__file__)
STRATEGY_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(STRATEGY_DIR)
sys.path.insert(0, PROJECT_ROOT)

print(f"📁 프로젝트 루트: {PROJECT_ROOT}\n")

from features.technical import TechnicalFeatures
from model.network import MAModel, Trainer


class MAStrategy:
    """
    이동평균 기반 트레이딩 전략
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        print("✅ 전략 초기화 완료")
    
    def generate_signals(self, features):
        """신호 생성"""
        print(f"\n📡 신호 생성 중...")
        
        with torch.no_grad():
            predictions = self.model(features)
            signals = torch.sign(predictions.squeeze())
        
        signals_np = signals.numpy()
        
        buy_count = (signals_np > 0).sum()
        sell_count = (signals_np < 0).sum()
        total = len(signals_np)
        
        print(f"   ✅ 총 {total}개 신호 생성")
        print(f"   📈 매수: {buy_count}개 ({buy_count/total*100:.1f}%)")
        print(f"   📉 매도: {sell_count}개 ({sell_count/total*100:.1f}%)")
        
        return signals_np
    
    def get_positions(self, signals):
        """포지션 계산"""
        print(f"\n🎯 포지션 계산 중...")
        
        positions = []
        current_position = 0  # 0=무포지션(현금), 1=롱, -1=숏
        
        for signal in signals:
            # === 수정: 첫 신호가 매도면 무시 ===
            if len(positions) == 0 and signal < 0:
                # 처음부터 매도 불가 (코인 없음)
                positions.append(0)
                continue
            
            # === 수정: 첫 신호가 매수면 바로 매수 ===
            if len(positions) == 0 and signal > 0:
                positions.append(1)
                current_position = 1
                continue
            
            # 매수 신호 + (무포지션 or 숏)
            if signal > 0 and current_position <= 0:
                positions.append(1)
                current_position = 1
            
            # 매도 신호 + (무포지션 or 롱)
            elif signal < 0 and current_position >= 0:
                positions.append(-1)
                current_position = -1
            
            # 같은 방향
            else:
                positions.append(0)
        
        positions = np.array(positions)
        
        buy_actions = (positions == 1).sum()
        sell_actions = (positions == -1).sum()
        hold_actions = (positions == 0).sum()
        
        print(f"   ✅ 포지션 계산 완료")
        print(f"   🔵 매수 실행: {buy_actions}회")
        print(f"   🔴 매도 실행: {sell_actions}회")
        print(f"   ⚪ 홀드: {hold_actions}회")
        
        return positions
    
    def analyze_signals(self, signals, prices, returns):
        """신호 품질 분석"""
        print(f"\n" + "=" * 60)
        print(f"📊 신호 품질 분석")
        print(f"=" * 60)
        
        buy_signal_returns = returns[signals > 0]
        sell_signal_returns = returns[signals < 0]
        
        print(f"\n매수 신호 성과:")
        print(f"   평균 수익률: {buy_signal_returns.mean():.6f}")
        print(f"   승률: {(buy_signal_returns > 0).mean()*100:.1f}%")
        
        print(f"\n매도 신호 성과:")
        print(f"   평균 수익률: {-sell_signal_returns.mean():.6f}")
        print(f"   승률: {(sell_signal_returns < 0).mean()*100:.1f}%")
        
        correct_predictions = (
            ((signals > 0) & (returns > 0)).sum() +
            ((signals < 0) & (returns < 0)).sum()
        )
        total = len(signals)
        accuracy = correct_predictions / total
        
        print(f"\n전체 정확도: {accuracy*100:.1f}%")
        print(f"=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 매매 전략 테스트 V0.1")
    print("=" * 60)
    
    # 데이터 경로
    data_path = os.path.join(PROJECT_ROOT, 'btc_1h_data.csv')
    
    print(f"\n📂 데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    
    tech = TechnicalFeatures(df)
    tech.add_moving_averages()
    tech.add_momentum_features()
    tech.add_labels()
    
    X, y = tech.get_features_and_labels()
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    print(f"✅ 데이터 준비 완료: {len(X)}개 샘플")
    
    # 모델 학습
    print(f"\n🔄 모델 학습 중...")
    model = MAModel(input_size=5)
    trainer = Trainer(model, lr=0.001)
    
    for epoch in range(200):
        loss = trainer.train_epoch(X_tensor, y_tensor)
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {loss:.6f}")
    
    print(f"✅ 학습 완료")
    
    # 전략 실행
    strategy = MAStrategy(model)
    signals = strategy.generate_signals(X_tensor)
    positions = strategy.get_positions(signals)
    
    # 분석
    prices = tech.df['close'].values
    returns = tech.df['future_return'].values
    strategy.analyze_signals(signals, prices, returns)
    
    # 결과 저장
    result_df = pd.DataFrame({
        'timestamp': tech.df['timestamp'].values,
        'price': prices,
        'signal': signals,
        'position': positions,
        'future_return': returns
    })
    
    result_path = os.path.join(PROJECT_ROOT, 'trading_signals.csv')
    result_df.to_csv(result_path, index=False)
    print(f"\n💾 신호 저장: {result_path}")
    
    print(f"\n✅ 전략 테스트 완료!")
    print(f"=" * 60)