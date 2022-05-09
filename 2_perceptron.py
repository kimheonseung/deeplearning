# Perceptron
# x1, x2 입력을 받아 w1, w2 가중치를 계산하여 y를 출력
# y값이 일정 임계값(theta) 기준으로 0 또는 1 출력
# y = 0 (w1x1 + w2x2 <= theta)
#     1 (w1x2 + w2x2 >  theta)

# AND (Linear)
# x1   x2   y
# 0    0    0
# 0    1    0
# 1    0    0
# 1    1    1
# => (w1, w2, theta) = (0.5, 0.5, 0.8), (1, 1, 1) ...
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# 수정된 AND (Linear)
# theta를 -b로 치환하여 임계값을 0으로 정규화
# y = 0 (w1x1 + w2x2 + b <= 0)
#     1 (w1x2 + w2x2 + b >  0)
import numpy as np
def ENHANCED_AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array((0.5, 0.5)) # 가중치
    b = -0.7                 # 편향
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# NAND (Linear)
# x1   x2   y
# 0    0    1
# 0    1    1
# 1    0    1
# 1    1    0
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array((0.5, 0.5))
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR (Linear)
# x1   x2   y
# 0    0    0
# 0    1    1
# 1    0    1
# 1    1    1
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array((0.5, 0.5))
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# XOR (Non Linear)
# x1   x2   y
# 0    0    0
# 0    1    1
# 1    0    1
# 1    1    0
# =>    NAND      OR
#    x1   x2   s1   s2   y
#    0    0    1    0    0
#    0    1    1    1    1
#    1    0    1    1    1
#    1    1    0    1    0
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# NAND에서 테트리스까지
# 이론상 2층 퍼셉트론이면 컴퓨터를 만들 수 있다.