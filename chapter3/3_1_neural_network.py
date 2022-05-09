"""
활성화 함수 (Activation Function)
입력 신호의 총합을 출력 신호로 변환하는 함수
"""


"""
계단함수
h(x) = 0 ( x <= 0 )
       1 ( x >  0 )
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
"""
import numpy as np
import matplotlib.pyplot as plt
def step_funtion(x):
    y = x > 0
    return y.astype(np.int0)
x = np.arange(-5.0, 5.0, 0.1)
y = step_funtion(x)
"""
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
"""

"""
시그모이드 함수
h(x) = 1 / ( 1 + exp(-x) )
"""
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
"""
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
"""

"""
ReLU 함수
입력이 0을 넘으면 입력 그대로를 출력, 0 이하면 0을 출력
h(x) = 0 ( x <= 0 )
       x ( x >  0 )
"""
def relu(x):
    return np.maximum(0, x)

"""
Identity 함수
입출력이 같은 함수
"""
def identity_function(x):
    return x



# 3층 신경망 구현
# 1층 (은닉층)
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
"""
print(W1.shape)
print(X.shape)
print(B1.shape)
"""
A1 = np.dot(X, W1) + B1
# 1층 정규화 (활성화 함수는 Sigmoid)
Z1 = sigmoid(A1)

# 2층 (은닉층)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
# 2층 정규화
Z2 = sigmoid(A2)

# 3층 (출력층)
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3

# 출력층의 활성화 함수는 identity function 사용
Y = identity_function(A3)

"""
출력층의 활성화 함수는 회귀인 경우 항등함수, 
2클래스 분류의 경우 시그모이드, 
다중클래스 분류에는 소프트맥스 함수를 사용
"""


# 구현 정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
# print(y)

"""
Softmax
y_k = exp(a_k) / ( sum(i = 1, n) exp(a_i) )
=> 총합은 1. 대소관계도 유지. 즉 출력을 '확률'로 해석할 수 있다.

Example
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)
print(np.sum(y))
"""
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
"""
위 함수는 Overflow 문제가 있음
개선된 소프트맥스
y_k = exp(a_k) / ( sum(i = 1, n) exp(a_i) )
    = C*exp(e_k) / C*( sum(i = 1, n) exp(a_i) )
    = exp(e_k + logC) / ( sum(i = 1, n) exp(a_i + logC) )
    = exp(e_k + C') / ( sum(i = 1, n) exp(a_i + C') )
상수처리를 통해 최댓값을 빼고 계산한다
"""
a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a))) # array([nan, nan, nan])
c = np.max(a)
print(np.exp(a-c) / np.sum(np.exp(a-c)))

def enhanced_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
