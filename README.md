# Test Theta

복소수 위상 기반 세타값 생성 및 테스트 폴더더

## 개요

AES-256 암호화와 복소수 연산을 통해 안전한 세타값(theta)을 생성하는 알고리즘입니다다. 
주파수 호핑과 그룹 순서 결정에 사용되는 위상값을 생성합니다.

## 주요 기능

- **복소수 위상 기반 정규화**: 황금비와 복소수 연산을 활용한 위상값 계산
- **AES-256 암호화**: 안전한 시드 생성을 위한 암호화 기능
- **Zadoff-Chu 시퀀스**: 연속 ZC 회전자를 활용한 고급 위상 계산
- **주파수 호핑**: 동적 주파수 할당을 위한 세타값 생성

## 파일 구조
```
├── README.md
├── requirements.txt
├── test-theta.py
└── theta_pkg
    ├── __init__.py
    └── generator.py
```
- `generator.py`: 세타값 생성
- `test-theta.py`: 생성된 세타값들의 랜덤성을 평가하기 위한 통계적 평가

## 세타값 생성성 함수

### `generate_theta(trial, indexOfTheta, versionOfComplexPhase)`

세타값을 생성하는 함수

**파라미터:**
- `trial`: 시도 횟수 (각 시도마다 다른 세타값 생성)
- `indexOfTheta`: 세타 인덱스
  - `1`: 그룹 순서를 결정하기 위한 세타값 생성
  - `2`: 주파수 위치를 결정하기 위한 세타값 생성
- `versionOfComplexPhase`: 복소수 위상 계산 버전
  - `3`: generate_complex_phase_V3 함수 사용 (7회 반복 연산)
  - `5`: generate_complex_phase_V5 함수 사용 (Zadoff-Chu 시퀀스 기반)

**반환값:**
- `float`: 계산된 위상값 (angle), -π에서 π 범위의 값

## 사용 방법

```python
from theta_pkg import generate_theta

# 그룹 순서용 세타값 생성 (버전 3)
theta1 = generate_theta(trial=0, indexOfTheta=1, versionOfComplexPhase=3)

# 주파수 위치용 세타값 생성 (버전 5)
theta2 = generate_theta(trial=0, indexOfTheta=2, versionOfComplexPhase=5)
```

## 의존성

- numpy
- matplotlib
- pycryptodome

## 설치

```bash
pip install numpy matplotlib pycryptodome
```
