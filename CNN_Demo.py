from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

#matplotlib inline
#config InlineBackend.figure_format = 'retina'

# 识别captcha验证码
# 1.导入captcha库：
import string
characters = string.digits
#characters = string.digits + string.ascii_uppercase
print(characters)

# 宽 x 高 x 字符数 x 字符种类
width, height, n_len, n_class = 170, 80, 4, len(characters)

# 2.定义数据生成器
from keras.utils.np_utils import to_categorical

def gen(batch_size=32):
    # X 的形状是 `(batch_size, height, width, 3)`，比如一批生成32个样本，图片宽度为170，高度为80，
    # 那么形状就是 `(32, 80, 170, 3)`，取第一张图就是 `X[0]`。
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

    # y 的形状是四个 `(batch_size, n_class)`，如果转换成 numpy 的格式，则是 `(n_len, batch_size, n_class)`，
    # 比如一批生成32个样本，  验证码的字符有36种，长度是4位，那么它的形状就是4个 `(32, 36)`，也可以说是 `(4, 32, 36)`
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X,y

# 测试生成器
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))
plt.show()

from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    # 将所有的Convolution2D替换为Conv2D
    # x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)

    # x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
#model = Model(input=input_tensor, output=x)
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# 可视化模型（略）

# 训练模型：
# model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
#                    validation_data=gen(), nb_val_samples=1280)

batch_size = 32
# 总数从51200调整为5120：nb_epoch为总轮数（共有几个epoch，即数据将被“轮”几次，1个epoch为1轮），
#model.fit_generator(gen(), steps_per_epoch=51200/batch_size, nb_epoch=5,
#                    validation_data=gen(), nb_val_samples=1280)
model.fit_generator(gen(), steps_per_epoch=5120/batch_size, nb_epoch=1,
                    validation_data=gen(), validation_steps=batch_size)

# 测试模型：
X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.axis('off')
plt.show()

# 计算模型整体准确率：
from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

evaluate(model)

# 保存模型
model.save('cnn.h5')
