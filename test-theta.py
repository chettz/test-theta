import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES 
import secrets
import sys
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 45  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.0225  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = SLOT_BANDWIDTH_MHZ * 17  # 그룹 하나당 대역폭 = 0.3825 MHz (0.0225 * 17 = 0.3825)
TOTAL_BANDWIDTH_MHZ = 200  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = SLOT_BANDWIDTH_MHZ * 5  # 최소 이격 거리 = 슬롯 기준 5개 = 0.0225 * 17 = 0.1125 MHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용

# === 복소수 위상 기반 정규화 ===
def generate_complex_phase(z: complex, iterations: int = 7):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)  # 위상값 계산 (phi를 이렇게 정의한 이유는 더욱 복잡하게 만들기 위해 z의 실수값과 황금비를 곱하고 그걸 0부터 1까지의 값으로 정규화함.)
        rotator = np.exp(1j * phi)  # 회전 연산자 (오일러 함수)
        z = (z*z + c) * rotator  # 회전 및 복소수 제곱 연산
    angle = np.angle(z)  # 복소수의 위상 (역연산으로 정확한 복소수 좌표를 얻기 힘듦. 해당 좌표와 원점이 이루는 선상의 모든 점의 좌표가 경우의 수가 되기 때문)
    norm = (angle + np.pi) / (2 * np.pi)  # 위상을 0~1 범위로 정규화 (위상을 그대로 사용할 경우에 값이 매우 커질 수 있음.)
    return norm, angle

# === 8바이트 시드로 복소수 생성 ===
def get_complex_from_seed(seed: bytes):
    real = int.from_bytes(seed[:4], 'big') / 1e9  # 상위 4바이트 -> 실수 (10^9로 나누는 이유는 unsigned일 떄, 최대 2^32-1 = 4,294,967,295 의 값을 가지기 때문에 10^9을 통해 값을 줄임.)
    imag = int.from_bytes(seed[4:8], 'big') / 1e9  # 하위 4바이트 → 허수 (,,,)
    return complex(real, imag)

# === AES 암호화 함수 (AES-256) ===
def aes_encrypt_block(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)  # AES 키는 32바이트 (AES-256)
    return cipher.encrypt(data)

# # === 주파수 시드 생성 함수 ===
# def generate_frequency_seed(tod, trial, group, key):  # 주파수 위치를 정하는 복소수 연산에 필요한 시드 생성 함수
#     pt = f"{tod}-{trial}-{group}".encode()  # 평문 구성 ( tod(송신 시각을 정밀하게 나타냄, 초 단위를 실수형으로 받아옴.) - 44비트 이내, 시도 횟수, 그룹 번호로 구성됨.f는 구분자)
#     pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
#     ct = aes_encrypt_block(pt_hash, key)  # AES 암호화
#     return get_complex_from_seed(ct)  # 초기 복소수 반환

# === 그룹 순서 시드 생성 함수 ===
def generate_group_order_seed(tod, trial, rank, key):  # 그룹 순서를 정하는 복소수 연산에 필요한 시드 생성 함수
    pt = f"{tod}-{trial}-R{rank}".encode()  # 순서용 평문 구성 ( tod(송신 시각을 정밀하게 나타냄) - 44비트, 시도 횟수, 랭크 값으로 구성됨. f, R은 구분자)
    pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
    ct = aes_encrypt_block(pt_hash, key)  # AES 암호화
    return get_complex_from_seed(ct[8:])  # 8~15바이트 사용

# === 세타값 생성 함수 ===
def generate_theta_1(trial):
    # key = b'0123456789abcdef0123456789abcdef'  # 32바이트 AES-256 키
    key = secrets.token_bytes(32)   # 랜덤 32바이트 키 생성
    tod = int(time.time()) & ((1 << 44) - 1) 
    for rank in range(GROUP_COUNT):
        z = generate_group_order_seed(tod, trial, rank, key)  # 그룹 순서를 결정하기 위한 복소수 시드 생성
        _, angle = generate_complex_phase(z)  # 위상값 계산
    return angle


# === 상관관계 종합 분석 함수 ===
def analyze_correlation(theta_values, max_lag=50, save_image=True):
    """
    theta 값들의 상관관계를 종합적으로 분석
    
    Args:
        theta_values: 분석할 theta 값들의 리스트
        max_lag: 분석할 최대 lag 값 (기본값: 50)
        save_image: 이미지 저장 여부 (기본값: True)
    
    Returns:
        acf_values: 자기상관함수 값들
        significant_lags: 통계적으로 유의한 lag들
    """
    n = len(theta_values)
    theta_array = np.array(theta_values)
    
    # 1. 자기상관함수(ACF) 계산
    acf_values = acf(theta_array, nlags=max_lag, fft=True)
    
    # 2. 신뢰구간 계산 (95% 신뢰구간)
    # 백색잡음의 경우 ±1.96/√n 범위 내에 있어야 함
    confidence_interval = 1.96 / np.sqrt(n)
    
    # 3. 통계적으로 유의한 lag 찾기(신뢰구간을 벗어나는)
    significant_lags = []
    for lag in range(1, max_lag + 1):
        if abs(acf_values[lag]) > confidence_interval:
            significant_lags.append((lag, acf_values[lag]))
    
    # 4. 시각화
    plt.figure(figsize=(15, 12))
    
    # 서브플롯 1: 자기상관함수 (ACF)
    plt.subplot(3, 2, 1)
    lags = np.arange(0, max_lag + 1)
    plt.plot(lags, acf_values, 'b-', linewidth=2, marker='o', markersize=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, 
                label=f'95% confidence interval: ±{confidence_interval:.4f}')
    plt.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 연속된 값들의 산점도 (lag=1)
    plt.subplot(3, 2, 2)
    plt.scatter(theta_array[:-1], theta_array[1:], alpha=0.5, s=1)
    plt.xlabel('θ(t)')
    plt.ylabel('θ(t+1)')
    plt.title('Scatter Plot: θ(t) vs θ(t+1)')
    plt.grid(True, alpha=0.3)
    
    # 상관계수 계산 및 표시
    corr_lag1 = np.corrcoef(theta_array[:-1], theta_array[1:])[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr_lag1:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 서브플롯 3: 다양한 lag에서의 산점도 (lag=50)
    plt.subplot(3, 2, 3)
    lags_to_plot = [50]
    colors = ['blue', 'green', 'orange']
    
    for i, lag in enumerate(lags_to_plot):
        if lag < len(theta_array):
            corr = np.corrcoef(theta_array[:-lag], theta_array[lag:])[0, 1]
            plt.scatter(theta_array[:-lag], theta_array[lag:], 
                       alpha=0.3, s=1, color=colors[i], 
                       label=f'Lag {lag}: r={corr:.4f}')
    
    plt.xlabel('θ(t)')
    plt.ylabel('θ(t+lag)')
    plt.title('Scatter Plots for Different Lags')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 시계열 플롯 (처음 100개 값)
    plt.subplot(3, 2, 4)
    plot_length = min(100, len(theta_array))
    plt.plot(range(plot_length), theta_array[:plot_length], 'b-', linewidth=1, marker='o', markersize=3)
    plt.title(f'Time Series Plot (First {plot_length} values)')
    plt.xlabel('Trial Index')
    plt.ylabel('Theta Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('theta_correlation_analysis.png', dpi=300, bbox_inches='tight')

    # 5. 결과 출력
    print("\n=== 상관관계 분석 결과 ===")
    print(f"총 샘플 수: {n}")
    print(f"분석한 최대 lag: {max_lag}")
    print(f"95% 신뢰구간: ±{confidence_interval:.6f}")
    
    print(f"\n주요 lag별 자기상관계수:")
    important_lags = [1, 2, 3, 5, 10, 20, 30, 50]
    for lag in important_lags:
        if lag <= max_lag:
            significance = "유의함" if abs(acf_values[lag]) > confidence_interval else "비유의"
            print(f"  Lag {lag:2d}: {acf_values[lag]:8.6f} ({significance})")
    
    print(f"\n통계적으로 유의한 lag 개수: {len(significant_lags)}")
    if significant_lags:
        print("유의한 lag들:")
        for lag, value in significant_lags[:10]:  # 처음 10개만 출력
            print(f"  Lag {lag:2d}: {value:8.6f}")
        if len(significant_lags) > 10:
            print(f"  ... 및 {len(significant_lags) - 10}개 더")
    
    # 6. 랜덤성 평가
    print(f"\n=== 랜덤성 평가 ===")
    
    # 유의한 상관관계의 비율
    significant_ratio = len(significant_lags) / max_lag * 100
    print(f"유의한 상관관계 비율: {significant_ratio:.2f}%")
    
    # 최대 절댓값 상관계수 (lag=0 제외)
    max_abs_corr = np.max(np.abs(acf_values[1:]))
    max_corr_lag = np.argmax(np.abs(acf_values[1:])) + 1
    print(f"최대 절댓값 상관계수: {max_abs_corr:.6f} (Lag {max_corr_lag})")
    
    # 평가 기준
    if significant_ratio < 5 and max_abs_corr < 0.1:
        randomness_assessment = "매우 좋음 (강한 랜덤성)"
    elif significant_ratio < 10 and max_abs_corr < 0.2:
        randomness_assessment = "좋음 (적절한 랜덤성)"
    elif significant_ratio < 20 and max_abs_corr < 0.3:
        randomness_assessment = "보통 (약간의 상관관계 존재)"
    else:
        randomness_assessment = "나쁨 (강한 상관관계 존재)"
    
    print(f"랜덤성 평가: {randomness_assessment}")

    
    # 7. 추가 독립성 검정
    print(f"\n=== 추가 독립성 검정 ===")
    
    # 제곱값들의 자기상관 (비선형 의존성 탐지)
    squared_values = theta_array ** 2
    acf_squared = acf(squared_values, nlags=min(20, max_lag), fft=True)
    max_squared_corr = np.max(np.abs(acf_squared[1:]))
    print(f"제곱값 최대 자기상관: {max_squared_corr:.6f}")
    
    # 절댓값들의 자기상관
    abs_values = np.abs(theta_array)
    acf_abs = acf(abs_values, nlags=min(20, max_lag), fft=True)
    max_abs_acf = np.max(np.abs(acf_abs[1:]))
    print(f"절댓값 최대 자기상관: {max_abs_acf:.6f}")
    
    return acf_values, significant_lags

# === 세타값 히스토그램 분석 함수 ===
def analyze_theta_histogram(theta_values, bins=50, save_image=True):
    """
    세타값들의 히스토그램을 생성하고 구간별 빈도의 통계를 분석
    
    Args:
        theta_values: 분석할 theta 값들의 리스트
        bins: 히스토그램 구간 수 (기본값: 50)
        save_image: 이미지 저장 여부 (기본값: True)
    
    Returns:
        hist: 각 구간의 빈도수 배열
        bins: 구간 경계값 배열
        std_dev: 구간별 빈도의 표준편차
    """
    # 히스토그램 데이터 계산
    hist, bin_edges = np.histogram(theta_values, bins=bins)
    
    # 통계 계산
    mean_freq = np.mean(hist)
    std_dev = np.std(hist)
    min_freq = np.min(hist)
    max_freq = np.max(hist)
    total_samples = len(theta_values)
    expected_freq = total_samples / bins  # 균등분포일 때 기대 빈도
    
    # 히스토그램 시각화
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 기본 히스토그램
    plt.subplot(2, 2, 1)
    plt.hist(theta_values, bins=bins, density=False, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, 
                label=f'Expected frequency (uniform): {expected_freq:.1f}')
    plt.title(f'Theta Values Histogram ({bins} bins)')
    plt.xlabel('Theta Value (radians)')
    plt.ylabel('Frequency Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 서브플롯 2: 구간별 빈도 막대그래프
    plt.subplot(2, 2, 2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(range(len(hist)), hist, alpha=0.7, color='orange', edgecolor='black')
    plt.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, 
                label=f'Expected frequency: {expected_freq:.1f}')
    plt.title('Frequency per Bin')
    plt.xlabel('Bin Index')
    plt.ylabel('Frequency Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('theta_histogram_analysis.png', dpi=300, bbox_inches='tight')
        
    # 결과 출력
    print("\n=== 히스토그램 분석 결과 ===")
    print(f"총 샘플 수: {total_samples}")
    print(f"구간 수: {bins}")
    print(f"구간별 평균 빈도: {mean_freq:.2f}")
    print(f"구간별 빈도의 표준편차: {std_dev:.4f}")
    print(f"최소 빈도: {min_freq}")
    print(f"최대 빈도: {max_freq}")
    print(f"빈도 범위: {max_freq - min_freq}")
    print(f"변동계수 (CV): {(std_dev/mean_freq)*100:.2f}%")
    print(f"균등분포 기대 빈도: {expected_freq:.2f}")
    
    # 균등성 평가
    cv = (std_dev/mean_freq)*100
    if cv < 5:
        uniformity_assessment = "매우 균등함"
    elif cv < 10:
        uniformity_assessment = "균등함"
    elif cv < 20:
        uniformity_assessment = "보통"
    else:
        uniformity_assessment = "불균등함"
    
    print(f"균등성 평가: {uniformity_assessment}")
    
    # 구간별 상세 정보 출력
    print(f"\n=== 구간별 상세 빈도 ===")
    for i in range(len(hist)):
        deviation = hist[i] - expected_freq
        print(f"구간 {i+1:2d}: [{bin_edges[i]:6.3f}, {bin_edges[i+1]:6.3f}] → "
              f"빈도: {hist[i]:4d}, 편차: {deviation:+6.2f}")
    
    return hist, bin_edges, std_dev

# === Kolmogorov-Smirnov 검정 함수 ===
def test_ks_uniformity(theta_values):
    """
    Kolmogorov-Smirnov 검정을 통해 theta 값들이 [-π, π] 구간의 균등분포를 따르는지 검정
    
    Args:
        theta_values: 검정할 theta 값들의 리스트
    
    Returns:
        ks_statistic: KS 통계량
        p_value: p-value
    """
    # theta 값들을 [0, 1] 구간으로 정규화 (균등분포 검정을 위해)
    normalized_values = (np.array(theta_values) + np.pi) / (2 * np.pi)
    
    # Kolmogorov-Smirnov 검정 수행 (균등분포와 비교)
    ks_statistic, p_value = stats.kstest(normalized_values, 'uniform')
    
    print("\n=== Kolmogorov-Smirnov 검정 결과 ===")
    print(f"KS 통계량: {ks_statistic:.6f}")
    print(f"p-value: {p_value:.6f}")
    
    # 결과 해석
    alpha = 0.05
    if p_value > alpha:
        print(f"결론: p-value ({p_value:.6f}) > α ({alpha}) → 균등분포를 따른다고 볼 수 있음 (귀무가설 채택)")
    else:
        print(f"결론: p-value ({p_value:.6f}) ≤ α ({alpha}) → 균등분포를 따르지 않음 (귀무가설 기각)")
    
    # 추가 통계 정보
    print(f"\n추가 정보:")
    print(f"- 샘플 수: {len(theta_values)}")
    print(f"- 원본 theta 범위: [{min(theta_values):.4f}, {max(theta_values):.4f}]")
    print(f"- 정규화된 값 범위: [{min(normalized_values):.4f}, {max(normalized_values):.4f}]")
    
    return ks_statistic, p_value

if __name__ == "__main__":
    trials = 10000  # 도약 횟수
    theta_values = []    # theta 값들을 저장할 리스트
    
    # theta 값 수집
    for trial in range(trials):
        theta = generate_theta_1(trial)
        theta_values.append(theta)
    
    # 상관관계 분석
    analyze_correlation(theta_values, max_lag=50, save_image=True)
    
    # # 히스토그램 분석
    # analyze_theta_histogram(theta_values, bins=50, save_image=True)
    
    # # KS 검정
    # test_ks_uniformity(theta_values)
 