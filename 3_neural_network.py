# 활성화 함수 (Activation Function)
# 입력 신호의 총합을 출력 신호로 변환하는 함수

# 계단함수
# h(x) = 0 ( x <= 0 )
#        1 ( x >  0 )
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0
import numpy as np
import matplotlib.pyplot as plt
def step_funtion(x):
    y = x > 0
    return y.astype(np.int)
x = np.arange(-5.0, 5.0, 0.1)
y = step_funtion(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드 함수
# h(x) = 1 / ( 1 + exp(-x) )
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# ReLU 함수
# 입력이 0을 넘으면 입력 그대로를 출력, 0 이하면 0을 출력
# h(x) = 0 ( x <= 0 )
#        x ( x >  0 )
def relu(x):
    return np.maximum(0, x)

# 3층 신경망 구현
