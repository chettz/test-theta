# Test Theta

복소수 위상 기반 세타값 생성 및 테스트 폴더

## 개요

AES-256 암호화와 복소수 연산을 통해 안전한 세타값(theta)을 생성하는 알고리즘입니다. 
주파수 호핑과 그룹 순서 결정에 사용되는 위상값을 생성합니다.

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
- `test-theta.py`: 생성된 세타값들의 랜덤성을 평가하기 위한 통계적 평가 제공

## 세타값 생성 함수(generator.py)

### `generate_theta(trial, indexOfTheta, versionOfComplexPhase)`

세타값을 생성하는 함수

**파라미터:**
- `trial`: 시도 횟수 (각 시도마다 다른 세타값 생성)
- `indexOfTheta`: 세타 인덱스
  - '1' $\theta_1$: 그룹 순서를 결정하기 위한 세타값 생성
  - '2' $\theta_2$: 주파수 위치를 결정하기 위한 세타값 생성
- `versionOfComplexPhase`: 복소수 위상 계산 버전
  - `3`: generate_complex_phase_V3 함수 사용 (7회 반복 연산)
  - `5`: generate_complex_phase_V5 함수 사용 (Zadoff-Chu 시퀀스 기반)

**반환값:**
- `float`: 계산된 위상값 (angle), -π에서 π 범위의 값

## 사용 방법

```python
from theta_pkg import generate_theta

# 그룹 순서용 세타값1 생성 (버전 3)
theta1 = generate_theta(trial=0, indexOfTheta=1, versionOfComplexPhase=3)

# 주파수 위치용 세타값2 생성 (버전 5)
theta2 = generate_theta(trial=0, indexOfTheta=2, versionOfComplexPhase=5)
```

## 세타값 통계적 평가 함수(test-theta.py)

`test-theta.py`에서 생성된 세타값들의 랜덤성과 균등성을 평가하기 위한 3가지 통계적 분석 함수를 제공합니다.

### 1. `analyze_correlation(theta_values, max_lag=50)`

**목적**: 세타값들 간의 시간적 상관관계를 분석하여 랜덤성을 평가

**기능**:
- **자기상관함수(ACF) 계산**: 연속된 세타값들 간의 상관관계 측정
- **신뢰구간 검정**: 95% 신뢰구간을 벗어나는 유의한 상관관계 탐지
- **시각화**: ACF 그래프, 산점도, 시계열 플롯 생성
- **랜덤성 평가**: 상관관계 비율과 최대 상관계수를 기반으로 랜덤성 등급 평가

**출력**: 
- 유의한 lag들의 개수와 값
- 랜덤성 평가 결과 (매우 좋음/좋음/보통/나쁨)
- 상관관계 분석 그래프 (`theta_correlation_analysis.png`)

### 2. `analyze_theta_histogram(theta_values, bins=50)`

**목적**: 세타값들의 분포가 균등한지 히스토그램을 통해 분석

**기능**:
- **히스토그램 생성**: 지정된 구간 수로 세타값 분포 시각화
- **균등성 검정**: 각 구간의 빈도가 균등분포 기대값과 얼마나 차이나는지 측정
- **변동계수 계산**: 구간별 빈도의 표준편차를 평균으로 나눈 값으로 균등성 평가
- **구간별 상세 분석**: 각 구간의 빈도와 기대값과의 편차 출력

**출력**:
- 구간별 빈도 통계 (평균, 표준편차, 최대/최소값)
- 균등성 평가 결과 (매우 균등함/균등함/보통/불균등함)
- 히스토그램 그래프 (`theta_histogram_analysis.png`)

### 3. `test_ks_uniformity(theta_values)`

**목적**: Kolmogorov-Smirnov 검정을 통해 세타값이 균등분포를 따르는지 통계적으로 검정

**기능**:
- **KS 검정 수행**: 세타값들을 [0,1] 구간으로 정규화 후 균등분포와 비교
- **p-value 계산**: 귀무가설(균등분포를 따름) 기각 여부 판단
- **통계적 결론**: α=0.05 기준으로 균등분포 여부 결정

**출력**:
- KS 통계량과 p-value
- 통계적 결론 (균등분포를 따름/따르지 않음)
- 원본 데이터와 정규화된 데이터의 범위 정보

### 사용 예시

```python
# 세타값 생성
theta_values = []
for trial in range(10000):
    theta = generate_theta(trial, 2, 5)
    theta_values.append(theta)

# 3가지 통계적 평가 수행
analyze_correlation(theta_values, max_lag=50)      # 상관관계 분석
analyze_theta_histogram(theta_values, bins=50)     # 히스토그램 분석  
test_ks_uniformity(theta_values)                   # KS 검정
```

## 의존성

- numpy
- matplotlib
- pycryptodome

```bash
pip install numpy matplotlib pycryptodome
```
