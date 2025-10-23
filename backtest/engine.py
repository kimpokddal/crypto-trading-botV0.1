import numpy as np
import pandas as pd

class Backtester:
    """백테스팅 엔진"""
    
    def __init__(self, initial_capital=10000, fee=0.001):
        self.initial_capital = initial_capital
        self.fee = fee
        
        print(f"💰 백테스팅 설정:")
        print(f"   초기 자본: ${initial_capital:,.2f}")
        print(f"   거래 수수료: {fee*100:.2f}%")
    
    def run(self, prices, positions):
        """백테스팅 실행"""
        print(f"\n🔄 백테스팅 시작...")
        print(f"   기간: {len(prices)}개 시간")
        
        capital = self.initial_capital
        holdings = 0
        equity_curve = [capital]
        trades = []
        current_state = 'cash'
        
        for i in range(len(positions)):
            price = prices[i]
            position = positions[i]
            
            # 매수: 현금 → 코인
            if position == 1 and current_state == 'cash':
                holdings = capital / price * (1 - self.fee)
                capital = 0
                current_state = 'holding'
                
                trades.append({
                    'index': i,
                    'type': 'BUY',
                    'price': price,
                    'amount': holdings
                })
                print(f"   🔵 매수: ${price:,.2f} (코인 {holdings:.4f}개)")
            
            # 매도: 코인 → 현금
            elif position == -1 and current_state == 'holding':
                capital = holdings * price * (1 - self.fee)
                holdings = 0
                current_state = 'cash'
                
                trades.append({
                    'index': i,
                    'type': 'SELL',
                    'price': price,
                    'amount': capital
                })
                print(f"   🔴 매도: ${price:,.2f} (현금 ${capital:,.2f})")
            
            # 현재 자산
            current_equity = capital + holdings * price
            equity_curve.append(current_equity)
        
        # 마지막 보유 중이면 매도
        if holdings > 0:
            final_price = prices[-1]
            capital = holdings * final_price * (1 - self.fee)
            holdings = 0
            
            trades.append({
                'index': len(prices) - 1,
                'type': 'SELL (Final)',
                'price': final_price,
                'amount': capital
            })
            print(f"   🔴 최종 매도: ${final_price:,.2f}")
        
        print(f"\n   ✅ 총 거래: {len(trades)}회")
        
        metrics = self.calculate_metrics(equity_curve, trades)
        
        return metrics, equity_curve, trades
    
    def calculate_metrics(self, equity_curve, trades):
        """성과 지표 계산"""
        print(f"\n📊 성과 지표 계산 중...")
        
        equity_array = np.array(equity_curve)
        final_capital = equity_array[-1]
        total_return = (final_capital / self.initial_capital - 1) * 100
        
        returns = np.diff(equity_array) / equity_array[:-1]
        
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760)
        else:
            sharpe_ratio = 0
        
        max_dd = self.max_drawdown(equity_array)
        
        # 승률 계산
        if len(trades) >= 2:
            profitable_trades = 0
            for i in range(1, len(trades)):
                if trades[i]['type'].startswith('SELL'):
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    
                    if sell_price > buy_price:
                        profitable_trades += 1
            
            num_pairs = len(trades) // 2
            win_rate = (profitable_trades / num_pairs) * 100 if num_pairs > 0 else 0
        else:
            win_rate = 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        print(f"   ✅ 계산 완료")
        
        return metrics
    
    def max_drawdown(self, equity_curve):
        """최대 낙폭"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd * 100
    
    def print_report(self, metrics):
        """성과 리포트"""
        print(f"\n" + "=" * 60)
        print(f"📈 백테스팅 결과 리포트")
        print(f"=" * 60)
        
        print(f"\n💰 자본:")
        print(f"   초기: ${metrics['initial_capital']:,.2f}")
        print(f"   최종: ${metrics['final_capital']:,.2f}")
        print(f"   변화: ${metrics['final_capital'] - metrics['initial_capital']:,.2f}")
        
        return_emoji = "📈" if metrics['total_return'] > 0 else "📉"
        print(f"\n{return_emoji} 수익률:")
        print(f"   총 수익률: {metrics['total_return']:.2f}%")
        
        print(f"\n⚠️  리스크:")
        print(f"   샤프 비율: {metrics['sharpe_ratio']:.2f}")
        print(f"   최대 낙폭: {metrics['max_drawdown']:.2f}%")
        
        print(f"\n🔄 거래:")
        print(f"   총 거래: {metrics['num_trades']}회")
        print(f"   승률: {metrics['win_rate']:.1f}%")
        
        print(f"\n📊 종합 평가:")
        
        if metrics['total_return'] > 10:
            print(f"   수익률: 우수 ✅")
        elif metrics['total_return'] > 0:
            print(f"   수익률: 양호 🟡")
        else:
            print(f"   수익률: 손실 ❌")
        
        if metrics['sharpe_ratio'] > 2:
            print(f"   샤프: 우수 ✅")
        elif metrics['sharpe_ratio'] > 1:
            print(f"   샤프: 양호 🟡")
        else:
            print(f"   샤프: 개선 필요 ❌")
        
        if metrics['max_drawdown'] < 10:
            print(f"   낙폭: 안전 ✅")
        elif metrics['max_drawdown'] < 20:
            print(f"   낙폭: 보통 🟡")
        else:
            print(f"   낙폭: 위험 ❌")
        
        print(f"=" * 60)