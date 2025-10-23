import sys
import os

print("=" * 60)
print("경로 테스트")
print("=" * 60)

# 현재 작업 디렉토리
print(f"\n현재 위치: {os.getcwd()}")

# Python 경로
print(f"\nPython 경로:")
for path in sys.path:
    print(f"  - {path}")

# 파일 존재 확인
print(f"\n파일 확인:")
print(f"  features/technical.py: {os.path.exists('features/technical.py')}")
print(f"  model/network.py: {os.path.exists('model/network.py')}")
print(f"  strategy/ma_strategy.py: {os.path.exists('strategy/ma_strategy.py')}")

# Import 테스트
print(f"\nImport 테스트:")
try:
    from features.technical import TechnicalFeatures
    print(f"  ✅ features.technical")
except Exception as e:
    print(f"  ❌ features.technical: {e}")

try:
    from model.network import MAModel
    print(f"  ✅ model.network")
except Exception as e:
    print(f"  ❌ model.network: {e}")

print("=" * 60)