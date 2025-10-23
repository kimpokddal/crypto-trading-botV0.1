import torch
import torch.nn as nn

class MAModel(nn.Module):
    """
    이동평균 기반 트레이딩 모델
    
    입력: [ma5, ma20, ma50, ma5_20_diff, ma20_50_diff]
    출력: 예측 수익률
    """
    
    def __init__(self, input_size=5):
        super().__init__()  # nn.Module 초기화 (필수!)
        
        # 레이어 정의
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),  # 5 → 32
            nn.ReLU(),                  # 음수 제거
            nn.Dropout(0.2),            # 과적합 방지
            
            nn.Linear(32, 16),          # 32 → 16
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1)            # 16 → 1 (수익률)
        )
    
    def forward(self, x):
        """
        순전파 (입력 → 출력)
        """
        return self.network(x)


class Trainer:
    """
    모델 학습 담당
    """
    
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr
        )
    
    def profit_loss(self, predictions, actual_returns):
        """
        커스텀 손실: 수익 최대화
        
        예측 부호와 실제 수익률 부호가 일치하면 보상
        """
        signals = torch.sign(predictions.squeeze())
        returns = signals * actual_returns
        return -returns.mean()
    
    def train_epoch(self, X, y):
        """
        1 에폭 학습
        """
        self.model.train()  # 학습 모드 (Dropout 작동)
        
        # Forward
        predictions = self.model(X)
        loss = self.profit_loss(predictions, y)
        
        # Backward
        self.optimizer.zero_grad()  # 기울기 초기화
        loss.backward()             # 기울기 계산
        self.optimizer.step()       # 파라미터 업데이트
        
        return loss.item()
    
    def evaluate(self, X, y):
        """
        평가 (Dropout 꺼짐)
        """
        self.model.eval()  # 평가 모드
        
        with torch.no_grad():  # 기울기 계산 안 함
            predictions = self.model(X)
            loss = self.profit_loss(predictions, y)
        
        return loss.item()


# 테스트 코드
if __name__ == "__main__":
    print("=== 모델 테스트 ===\n")
    
    # 가짜 데이터
    X = torch.randn(100, 5)  # 100개 샘플, 5개 피처
    y = torch.randn(100)     # 100개 레이블
    
    # 모델 생성
    model = MAModel(input_size=5)
    print("모델 구조:")
    print(model)
    print()
    
    # 학습
    trainer = Trainer(model, lr=0.001)
    
    print("학습 시작:")
    for epoch in range(5):
        loss = trainer.train_epoch(X, y)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 평가
    eval_loss = trainer.evaluate(X, y)
    print(f"\n평가 Loss: {eval_loss:.4f}")