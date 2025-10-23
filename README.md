# 암호화폐 트레이딩 봇 V0.1

이동평균 기반 딥러닝 트레이딩 봇 MVP

## 특징
- PyTorch 신경망
- 이동평균 전략 (MA5, MA20, MA50)
- 백테스팅 엔진
- ccxt API 연동

## 설치
```bash
pip install torch ccxt pandas numpy
```

## 실행
```bash
python main.py
```

## 경고 ⚠️
**V0.1은 학습용 MVP입니다.**  
**실제 거래에 절대 사용하지 마세요!**

현재 성과: -6.86% 손실

## 다음 버전
- V0.2: 더 많은 데이터 & 피처
- V0.3: LSTM & 리스크 관리
- V0.4: 페이퍼 트레이딩
