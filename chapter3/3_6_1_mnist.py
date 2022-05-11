import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
"""
normalize     : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이로 정규화
flatten       : 입력 이미지를 평탄하게, 즉 1차원으로 만듬
                False인 경우 1 X 28 X 28 의 3차원 배열로,
                True인 경우 784개의 원소로 이루어진 1차원 배열로 저장
one_hot_label : 정답을 뜻하는 원소만 1이고 나머지는 모두 0인 배열로 저장
"""

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]

print(img.shape) # (784, )
img = img.reshape(28, 28)
print(img.shape) # (28, 28)
img_show(img)
