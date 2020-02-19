#
# 参考https://blog.csdn.net/javastart/article/details/88087966，同过CNN方式实现端到端的验证码OCR识别。
#
# 1. 修改为Keras实现：其中验证码数据用自己训练前生成的方式；
# 2. 新增自定义accuracy计算，判断识别完全成功的比例；
#
# 目前识别比例可达92%！


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import numpy as np
import cv2, os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from ImagePreprocess import genCaptcha, codeConvert, recordTrainingHistory
from keras.preprocessing.image import ImageDataGenerator

class CnnOcr:
    def __init__(self):
        self.epochs = 5                                     # 最大迭代epoch次数
        self.batch_size = 64                                # 训练时每个批次参与训练的图像数目，显存不足的可以调小
        self.lr = 1e-3*0.2                                  # 初始学习率
        self.save_epoch = 1                                 # 每相隔多少个epoch保存一次模型
        self.im_width = 128
        self.im_height = 64
        self.im_total_num = 10000                           # 总共生成的验证码图片数量
        self.train_max_num = self.im_total_num              # 训练时读取的最大图片数目
        self.val_size = 0.1                                 # 用于验证的比例
        self.val_num = self.im_total_num * self.val_size    # 不能大于self.train_max_num  做验证集用
        # self.val_num = 50 * self.batch_size                 # 不能大于self.train_max_num  做验证集用
        self.words_num = 4                                  # 每张验证码图片上的数字个数
        self.words = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.label_num = self.words_num * len(self.words)   # one-hot
        self.keep_drop = 0.5
        self.x = None
        self.y = None
        self.model_path = os.path.join(os.getcwd(), 'keras_cnn_ocr_model.h5')

    # 自定义成功率函数accuracy:
    def mean_pred(self, y_true, y_pred):
        # Keras的Reshape不需要带-1的batch_size：predict的shape为(None, 4, 26)
        predict = Reshape([self.words_num, len(self.words)])(y_pred)
        max_idx_p = tf.argmax(predict, 2)       # 2代表对三维数组按列求最大值所在位置

        y_predict = Reshape([self.words_num, len(self.words)])(y_true)
        max_idx_l = tf.argmax(y_predict, 2)

        # correct_pred返回为[none, 4]:
        correct_pred = tf.equal(max_idx_p, max_idx_l)

        # tf.cast将tf.equal逐位比较的True/False结果转换为浮点型（确保取均值时为有小数位），tf.prod按行相乘（全对才为1），tf.reduce_mean求得均值：
        accuracy = tf.reduce_mean(tf.reduce_prod(tf.cast(correct_pred, tf.float32), axis=1))

        return accuracy

    def pred_oneimage(self, image, model=None):

        if model == None:
            print("Load model from： ", self.model_path)
            model = load_model(self.model_path, custom_objects={'mean_pred': self.mean_pred})

        img_data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, dsize=(self.im_width, self.im_height))
        img_data = img_data.reshape(1, self.im_height, self.im_width, 1).astype(np.uint8)

        img_data -= 147
        # cv2.imshow("PREDICT", img_data[0])
        # cv2.waitKey(2000)

        self.keep_drop = 1.0
        pred = model.predict(img_data, batch_size=1)
        codeCvt = codeConvert(self.words)

        return codeCvt.onehot2str(pred[0])


    def train(self):
        # input_shape = (self.im_height, self.im_width, 1)

        x, y = genCaptcha(self.words, capt_len=self.words_num, width=self.im_width, height=self.im_height, img_num=self.im_total_num, color_mode="grayscale")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        # 增加：去均值化
        x_train -= 147
        x_test -= 147

        # 定义图片预处理器：图像增强
        img_gen = ImageDataGenerator(
            # rescale=1. / 255,
            # rotation_range=5,
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # horizontal_flip=False,
            # vertical_flip=False,
            # fill_mode='nearest',
            # preprocessing_function=None,
            # fill_mode='wrap', zoom_range=[2, 2],
            # data_format='channels_last'
        )

        train_gen = img_gen.flow(x_train, y_train, batch_size=self.batch_size,
                                 # save_to_dir=os.path.join(os.getcwd(), "data"),
                                 # save_prefix='F',
                                 # save_format='jpeg'
                                 )

        dat = train_gen.next()
        cv2.imshow("DAT", dat[0][0])
        cv2.waitKey(2000)
        print(dat[1][0])

        try:
            print("Load model: ", self.model_path)
            # 包含自定义loss，accuracy等时，装载模型时要加入custom_objects!
            model = load_model(self.model_path, custom_objects={'mean_pred': self.mean_pred})
            print("Load success!")

        except Exception as e:
            print("Error: ", e)
            print("Now init new model ...")
            model = self.cnnNet()

        finally:
            # 指定adam的LR?
            model.compile(
                loss=keras.losses.binary_crossentropy,
                # optimizer='adam',
                optimizer=Adam(lr=self.lr, beta_1=0.5),
                metrics=['accuracy', self.mean_pred])

            H = model.fit_generator(train_gen, steps_per_epoch=len(x_train) / self.batch_size, epochs=self.epochs,
                                    verbose=1, validation_data=(x_test, y_test))

            print("Save model to: ", self.model_path)
            model.save(self.model_path)
            recordTrainingHistory(H, epochs=self.epochs)

        # lr = self.lr * (1 - (epoch_num / self.epoch_max) ** 2)  # 动态学习率


    # CNN网络：
    def cnnNet(self):
        model = Sequential()

        # print("MODEL(" + model_path + "): captcha_len=" + str(captcha_len) + " category=" + str(num_classes))

        # input_shape = (64, 128, 3)
        input_shape = (self.im_height, self.im_width, 1)

        # 注意：若不加“padding='same'”, 会导致输出减少，从64降为60！往后都会递减。
        # conv1：Tensor("Relu:0", shape=(None, 64, 128, 32), dtype=float32)
        # net = tf.nn.conv2d(self.x, [5, 5, 3, 32], [1, 1, 1, 1], 'SAME')  3代表什么输入？
        # model.add(Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(self.lr), input_shape=input_shape))
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))

        # pool1：Tensor("MaxPool:0", shape=(None, 32, 64, 32), dtype=float32)
        # tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2)))
        # model.add(Dropout(0.1))
        # model.add(BatchNormalization())

        # 2 conv
        # conv2：Tensor("Relu_1:0", shape=(None, 32, 64, 64), dtype=float32)
        # net = tf.nn.conv2d(net, [5, 5, 32, 64], [1, 1, 1, 1], padding='SAME')  # 卷积
        model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

        # pool2：Tensor("MaxPool_1:0", shape=(None, 16, 32, 64), dtype=float32)
        # net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2)))
        # model.add(Dropout(0.1))
        # model.add(BatchNormalization())

        # 3 conv
        # conv3：Tensor("Relu_2:0", shape=(None, 16, 32, 64), dtype=float32)
        # net = tf.nn.conv2d(net, [3, 3, 64, 64], [1, 1, 1, 1], padding='SAME')  # 卷积
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # pool3：Tensor("MaxPool_2:0", shape=(None, 8, 16, 64), dtype=float32)
        # net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2)))
        # model.add(Dropout(0.1))
        # model.add(BatchNormalization())

        # 4 conv
        # conv4：Tensor("Relu_3:0", shape=(None, 8, 16, 64), dtype=float32)
        # net = tf.nn.conv2d(net, [3, 3, 64, 64], [1, 1, 1, 1], padding='SAME')  # 卷积
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # pool4：Tensor("MaxPool_3:0", shape=(None, 4, 8, 64), dtype=float32)            ？？？没有
        # net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化
        # model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(1, 1)))      # [4,8,64],输出为为[7,15,64]，导致模型大小将接近翻倍
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2)))
        # model.add(Dropout(0.1))
        # model.add(BatchNormalization())

        # 拉伸flatten，把多个图片同时分别拉伸成一条向量
        # 拉伸flatten：Tensor("Reshape:0", shape=(None, 2048), dtype=float32)
        model.add(Flatten())

        # 1 Dense
        # fc第一层：Tensor("Relu_4:0", shape=(None, 2048), dtype=float32)
        # net = tf.matmul(net, [8 * 4 * 64, 2048])
        model.add(Dense(2048, activation='relu', ))
        # model.add(BatchNormalization())
        model.add(Dropout(self.keep_drop))

        # fc第二层：Tensor("Relu_5:0", shape=(None, 2048), dtype=float32)
        # net = tf.matmul(net, [2048, 2048])
        model.add(Dense(2048, activation='relu', ))
        model.add(Dropout(self.keep_drop))

        # 2 Dense
        # net = tf.matmul(net, [2048, self.label_num])
        # input_shape, captcha_len = 5, num_classes = 36, model_path = "my_model.h5"
        # model.add(Dense(self.label_num, activation='softmax'))
        model.add(Dense(self.label_num, activation='sigmoid'))            # ???

        # sigmoid+交叉熵函数, 在tensorflow上体现为sigmoid_cross_entropy_with_logits, 在Keras上体现为binary_crossentropy
        # softmax+交叉熵函数, 在tensorflow上体现为softmax_cross_entropy_with_logits_v2, 在Keras中对应的是categorical_crossentropy
        # sparse+softmax+交叉熵函数：在tensorflows中体现为sparse_softmax_cross_entropy_with_logits, 在keras中对应的为sparse_categorical_crossentropy
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


        model.summary()
        # model.save(self.model_path)
        # print("Save model to: ", self.model_path)

        return model


if __name__ == '__main__':
    opt_type = input("Please choose: 1.train  2.test ")

    instance = CnnOcr()

    if opt_type == '1':
        instance.train()

    elif opt_type == '2':
        model = load_model(instance.model_path, custom_objects={'mean_pred': instance.mean_pred})
        # print(instance.pred_oneimage('./data/BKVR_7.jpg', model=model))
        # print(instance.pred_oneimage('./data/CXYS_6.jpg', model=model))
        # print(instance.pred_oneimage('./data/IJJS_9.jpg', model=model))
        imgpath = os.path.join(os.getcwd(), './data/test')
        imgs = os.listdir(imgpath)
        wrong = 0
        for img in imgs:
            lable = instance.pred_oneimage(os.path.join(imgpath, img), model=model)
            real = img.split("_")[1][0:4]
            if lable != real:
                print(real, lable)
                wrong += 1
        print("Accurancy: ", (len(imgs)-wrong)/len(imgs))
