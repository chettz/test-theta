import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES 
import secrets
import sys

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 45  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.0225  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = SLOT_BANDWIDTH_MHZ * 17  # 그룹 하나당 대역폭 = 0.3825 MHz (0.0225 * 17 = 0.3825)
TOTAL_BANDWIDTH_MHZ = 200  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = 0.1
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
# 16진수 문자열을 바이너리로 변환하여 32바이트 AES-256 키 생성
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')


# === 복소수 위상 기반 정규화_V3 ===
def generate_complex_phase_V3(z: complex, iterations: int = 7):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)  # 위상값 계산 (phi를 이렇게 정의한 이유는 더욱 복잡하게 만들기 위해 z의 실수값과 황금비를 곱하고 그걸 0부터 1까지의 값으로 정규화함.)
        rotator = np.exp(1j * phi)  # 회전 연산자 (오일러 함수)
        z = (z*z + c) * rotator  # 회전 및 복소수 제곱 연산
    angle = np.angle(z)  # 복소수의 위상 (역연산으로 정확한 복소수 좌표를 얻기 힘듦. 해당 좌표와 원점이 이루는 선상의 모든 점의 좌표가 경우의 수가 되기 때문)
    norm = (angle + np.pi) / (2 * np.pi)  # 위상을 0~1 범위로 정규화 (위상을 그대로 사용할 경우에 값이 매우 커질 수 있음.)
    return norm, angle

# === 복소수 위상 반복 함수_V5 ===
def generate_complex_phase_V5(z: complex, iterations: int = 1):
    N = 839  # ZC 시퀀스 길이
    u = 25   # ZC 루트

    GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

    # 연속 Zadoff–Chu 회전자 생성
    index = (abs(z) * GOLDEN_RATIO * N) % N  # 실수 기반 연속 인덱스
    phi_zc = -np.pi * u * index * (index + 1) / N
    rotator_zc = np.exp(1j * phi_zc)

    # 복소 반복 수식
    for _ in range(iterations):
        phi = 2 * np.pi * ((abs(z) * GOLDEN_RATIO) % 1)
        rotator = np.exp(1j * phi)
        z = (z * z + c) * rotator * rotator_zc  # 연속 ZC 보조 회전 적용

    # 위상 정규화
    angle = np.angle(z)
    norm = (angle + np.pi) / (2 * np.pi)

    return norm, angle

# === 8바이트 시드로 복소수 생성 ===
def get_complex_from_seed(seed: bytes):
    real = int.from_bytes(seed[:4], 'big') / 1e9  # 상위 4바이트 -> 실수 (10^9로 나누는 이유는 unsigned일 떄, 최대 2^32-1 = 4,294,967,295 의 값을 가지기 때문에 10^9을 통해 값을 줄임.)
    imag = int.from_bytes(seed[4:8], 'big') / 1e9  # 하위 4바이트 → 허수 (,,,)
    return complex(real, imag)

# === AES-256 암호화 함수 ===
def aes_encrypt_block(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)  # AES 키는 32바이트 (AES-256)
    return cipher.encrypt(data)

# === 주파수 시드 생성 함수 ===
def generate_frequency_seed(tod, trial, group, key):  # 주파수 위치를 정하는 복소수 연산에 필요한 시드 생성 함수
    pt = f"{tod}-{trial}-{group}".encode()  # 평문 구성 ( tod(송신 시각을 정밀하게 나타냄, 초 단위를 실수형으로 받아옴.) - 44비트 이내, 시도 횟수, 그룹 번호로 구성됨.f는 구분자)
    pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
    ct = aes_encrypt_block(pt_hash, key)  # AES 암호화
    return get_complex_from_seed(ct)  # 초기 복소수 반환

# === 그룹 순서 시드 생성 함수 ===
def generate_group_order_seed(tod, trial, rank, key):  # 그룹 순서를 정하는 복소수 연산에 필요한 시드 생성 함수
    pt = f"{tod}-{trial}-{rank}".encode()  # 순서용 평문 구성 ( tod(송신 시각을 정밀하게 나타냄) - 44비트, 시도 횟수, 랭크 값으로 구성됨. f, R은 구분자)
    pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
    ct = aes_encrypt_block(pt_hash, key)  # AES 암호화
    return get_complex_from_seed(ct[8:])  # 8~15바이트 사용

# === 세타값 생성 함수 ===
def generate_theta(trial: int, indexOfTheta: int, versionOfComplexPhase: int) -> float:
    """
    위상상값(theta)을 생성하는 함수
    
    
    trial (int): 시도 횟수. 각 시도마다 다른 세타값을 생성하기 위한 파라미터
    indexOfTheta (int): 세타 인덱스
        - indexOfTheta = 1: 그룹 순서를 결정하기 위한 세타값 생성
        - indexOfTheta = 2: 주파수 위치를 결정하기 위한 세타값 생성
    
    versionOfComplexPhase (int): 복소수 위상 계산 version
        - versionOfComplexPhase = 3: generate_complex_phase_V3 함수 사용 (7회 반복 연산)
        - versionOfComplexPhase = 5: generate_complex_phase_V5 함수 사용 (Zadoff-Chu 시퀀스 기반)
    
    customTOD : 사용자 지정 tod 값. None이면 현재 시간 사용
    
    Returns:
        float: 계산된 위상값 (angle), -π에서 π 범위의 값
    """

    tod = int(time.time()) & ((1 << 44) - 1)
        
    if indexOfTheta == 1:
        for rank in range(GROUP_COUNT):
            z = generate_group_order_seed(tod, trial, rank, AES_KEY) 
            if versionOfComplexPhase == 3:
                _, angle = generate_complex_phase_V3(z)  # 위상값 계산 버전 3 적용
            elif versionOfComplexPhase == 5:
                _, angle = generate_complex_phase_V5(z)  # 위상값 계산 버전 5 적용
    elif indexOfTheta == 2:
        for group_id in range(GROUP_COUNT):
            z = generate_frequency_seed(tod, trial, group_id, AES_KEY)  
            if versionOfComplexPhase == 3:
                _, angle = generate_complex_phase_V3(z)  # 위상값 계산 버전 3 적용
            elif versionOfComplexPhase == 5:
                _, angle = generate_complex_phase_V5(z)  # 위상값 계산 버전 5 적용
    return angle


if __name__ == "__main__":
    trial = 0
    hex_str = AES_KEY.hex().upper()
    print(hex_str)

    # while True:
    #     theta = generate_theta(trial, 2, 5)
    #     # angle 값(-π에서 π)을 0~2^32-1 사이의 정수로 변환
    #     # 1. angle을 0~2π 범위로 변환 (angle + π)
    #     # 2. 0~1 범위로 정규화 ((angle + π) / (2π))
    #     # 3. 32비트 정수 범위로 스케일링
    #     integer_value = int(((theta + np.pi) / (2 * np.pi)) * (2**32 - 1))
    #     print(f"Trial {trial}: {integer_value}")
    #     trial += 1
