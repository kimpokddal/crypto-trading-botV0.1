"""
암호화폐 트레이딩 봇 V0.1
이동평균 기반 딥러닝 전략
"""

import torch
import pandas as pd
import os
import sys

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# === 정상 import ===
from data.collector import DataCollector
from features.technical import TechnicalFeatures
from model.network import MAModel, Trainer
from strategy.ma_strategy import MAStrategy
from backtest.engine import Backtester


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🤖 암호화폐 트레이딩 봇 V0.1")
    print(" " * 20 + "이동평균 딥러닝 전략")
    print("=" * 70)


def print_section(title):
    """섹션 헤더"""
    print(f"\n{'─' * 70}")
    print(f"📍 {title}")
    print(f"{'─' * 70}")


def main():
    """메인 실행 함수"""
    
    print_header()
    
    # 설정
    print_section("1. 설정")
    
    config = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'limit': 1000,
        'ma_periods': [5, 20, 50],
        'epochs': 200,
        'learning_rate': 0.001,
        'initial_capital': 10000,
        'fee': 0.001
    }
    
    print(f"   거래소: {config['exchange']}")
    print(f"   심볼: {config['symbol']}")
    print(f"   시간봉: {config['timeframe']}")
    print(f"   데이터: {config['limit']}개")
    print(f"   이동평균: {config['ma_periods']}")
    print(f"   에폭: {config['epochs']}")
    print(f"   학습률: {config['learning_rate']}")
    print(f"   초기 자본: ${config['initial_capital']:,}")
    print(f"   수수료: {config['fee']*100}%")
    
    # 데이터 수집
    print_section("2. 데이터 수집")
    
    collector = DataCollector(
        exchange=config['exchange'],
        symbol=config['symbol'],
        timeframe=config['timeframe']
    )
    
    df = collector.fetch_ohlcv(limit=config['limit'])
    
    # 피처 생성
    print_section("3. 피처 생성")
    
    tech = TechnicalFeatures(df)
    tech.add_moving_averages(periods=config['ma_periods'])
    tech.add_momentum_features()
    tech.add_labels()
    
    X, y = tech.get_features_and_labels()
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    print(f"✅ 학습 데이터 준비: X={X.shape}, y={y.shape}")
    
    # 모델 학습
    print_section("4. 모델 학습")
    
    model = MAModel(input_size=5)
    trainer = Trainer(model, lr=config['learning_rate'])
    
    print(f"🔄 {config['epochs']} 에폭 학습 시작...")
    
    for epoch in range(config['epochs']):
        loss = trainer.train_epoch(X_tensor, y_tensor)
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch}/{config['epochs']}: Loss = {loss:.6f}")
    
    print(f"✅ 학습 완료")
    
    # 모델 저장
    model_path = os.path.join(PROJECT_ROOT, 'model_v0.1.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 모델 저장: {model_path}")
    
    # 신호 생성
    print_section("5. 매매 신호 생성")
    
    strategy = MAStrategy(model)
    signals = strategy.generate_signals(X_tensor)
    positions = strategy.get_positions(signals)
    
    # 신호 품질 분석
    prices = tech.df['close'].values
    returns = tech.df['future_return'].values
    strategy.analyze_signals(signals, prices, returns)
    
    # 신호 저장
    signals_df = pd.DataFrame({
        'timestamp': tech.df['timestamp'].values,
        'price': prices,
        'signal': signals,
        'position': positions,
        'future_return': returns
    })
    
    signals_path = os.path.join(PROJECT_ROOT, 'trading_signals.csv')
    signals_df.to_csv(signals_path, index=False)
    print(f"\n💾 신호 저장: {signals_path}")
    
    # 백테스팅
    print_section("6. 백테스팅")
    
    backtester = Backtester(
        initial_capital=config['initial_capital'],
        fee=config['fee']
    )
    
    metrics, equity_curve, trades = backtester.run(prices, positions)
    backtester.print_report(metrics)
    
    # 자산 곡선 저장
    equity_df = pd.DataFrame({'equity': equity_curve})
    equity_path = os.path.join(PROJECT_ROOT, 'equity_curve.csv')
    equity_df.to_csv(equity_path, index=False)
    print(f"\n💾 자산 곡선 저장: {equity_path}")
    
    # 요약
    print_section("7. 최종 요약")
    
    print(f"\n📊 V0.1 성과:")
    print(f"   데이터 기간: {tech.df['timestamp'].min()} ~ {tech.df['timestamp'].max()}")
    print(f"   샘플 수: {len(X)}개")
    print(f"   거래 횟수: {metrics['num_trades']}회")
    print(f"   총 수익률: {metrics['total_return']:.2f}%")
    print(f"   샤프 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"   최대 낙폭: {metrics['max_drawdown']:.2f}%")
    
    print(f"\n💾 생성된 파일:")
    print(f"   1. model_v0.1.pth")
    print(f"   2. trading_signals.csv")
    print(f"   3. equity_curve.csv")
    
    print(f"\n⚠️  경고:")
    print(f"   V0.1은 학습용 MVP입니다.")
    print(f"   실제 거래에 절대 사용하지 마세요!")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "✅ V0.1 완성!")
    print("=" * 70)
    
    print(f"\n🎯 다음 단계:")
    print(f"   1. V0.2: 더 많은 데이터")
    print(f"   2. V0.2: 더 많은 피처")
    print(f"   3. V0.3: LSTM")
    print(f"   4. V0.4: 페이퍼 트레이딩")
    
    print(f"\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()