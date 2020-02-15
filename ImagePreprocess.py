#
# 图片数据预处理和生成器：数据增强，避免一次装入所以数据到内存！
#
# 注意：在本机上运行可以，在Google Colab上运行速度非常慢（涉及磁盘读写都比较慢，最好一次读入！）
#

import os, cv2, string, random
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt

# 用LabelBinarizer实现标签二值化：
# label = ['yes', 'no', 'yes', 'no', 'none']
# lb = preprocessing.LabelBinarizer() #构建一个转换对象
# Y = lb.fit_transform(label)
# re_label = lb.inverse_transform(Y)
# print(Y, re_label)

# # 全部解
# print(np.argmax(encoded_data, axis=1))


# 记录训练历史：
def recordTrainingHistory(H, epochs=10, img_save="plot.png"):
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(img_save)


# 定义图片预处理器：
data_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    # horizontal_flip=False,
    # vertical_flip=False,
    fill_mode='nearest',
    # fill_mode='wrap', zoom_range=[2, 2],
    # data_format='channels_last'
    )

# x = cv2.imread('train/0001.jpg')
# x = x.reshape((1,) + x.shape)                           # 要求为4维

# 方式1：使用flow()，flow可采用x_train, y_train作为输入

# for batch in data_gen.flow(x, batch_size=1,
#                            save_to_dir='/temp',
#                            save_prefix='flower',
#                            save_format='jpeg'):
# 方式2：使用flow_from_directory
# gen_data = data_gen.flow_from_directory(
#     "../DATA/Captcha/testing",
#     batch_size=1,
#     shuffle=False,
#     save_to_dir='/temp',
#     save_prefix='gend',
#     save_format='jpeg',
#     target_size=(600, 600))
#
# for i in range(2):
#    img = gen_data.next()


# 方式3：使用自定义生成器

# 标签和One-hot的相互转换：
class codeConvert(object):
    def __init__(self, charactors):
        self.char = charactors
        self.category = len(charactors)
        self.encoder = LabelEncoder().fit(np.array([str(x) for x in charactors]))

    # 将字符串转换为对应的one-hot编码：
    def str2onehot(self, lablestr):
        # 先字符转换为数字编码：
        numcode = self.encoder.transform([x for x in lablestr])
        # 再将数字转为one-hot编码：返回拼接后的numpy数组
        return np.hstack(to_categorical(numcode, self.category))

    def onehot2str(self, onehot):
        char_len = int(len(onehot)/self.category)
        lablestr = ""

        for i in range(char_len):
            onehot_part = onehot[self.category*i:self.category*(i+1)]
            # 将onehot编码转换为数字：
            numcode = np.argmax(onehot_part)
            # 将数字编码转换为对应的字符：
            lablestr += self.encoder.inverse_transform([numcode])[0]
        return lablestr


# 文件名代表标签：其中文件名及编码由charactors组成，文件名形如xxxx_yyyy.zzz, xxxx被视作标签。
# 当为灰度模式时，进行图像二值化处理：
def imageWithLable_generator(imagePath, bs=32, mode="train", aug=None,
                             charactors=string.digits+string.ascii_uppercase,
                             color_mode='grayscale',
                             shffle=True
                             ):

    # listdir返回值只包含文件名：
    img_file = os.listdir(imagePath)
    if shffle:
        random.seed(42)
        random.shuffle(img_file)        # 乱序！
    print("Total image number=", len(img_file))

    # one-hot encode the labels
    codeCvt = codeConvert(charactors)

    # loop indefinitely
    i = 0
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []

        # 生成一个批次的数据：
        while len(images) < bs:
            if len(img_file) == 0:
                break

            os.path.join(imagePath, img_file[i])

            if color_mode == 'grayscale':
                image = cv2.imread(os.path.join(imagePath, img_file[i]), cv2.IMREAD_GRAYSCALE)
                # Otsu二值化阈值：对双峰图像自动计算阈值，传入参数增加cv2.THRESH_OTSU，
                # 并把阈值设为0, 返回值retVal就是最优阈值
                ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                image = image.reshape(image.shape + (1,))           # 转为3维
            else:
                image = cv2.imread(os.path.join(imagePath, img_file[i]))

            images.append(image)
            labels.append(codeCvt.str2onehot(img_file[i].split("_")[0]))

            # check to see if the line is empty, indicating we have reached the end of the file
            if i == len(img_file)-1:
                # reset the file pointer to the beginning of the file and re-read the line
                print("End!!! reset to begin ...")
                i = 0

                # 验证数据时：只读取一次。
                if mode == "eval":
                    print("break")
                    # break
                    return
            else:
                i += 1

        # 数据增强：
        if aug is not None:
            images, labels = next(aug.flow(np.array(images), labels, batch_size=bs))

        # 生成器：yield the batch to the calling function
        yield np.array(images), labels


# 采用Captcha生成验证码：倾斜，位置移动，变形等
# -characters：为生产验证码的字符集，缺省为数字+大写字母
# -color_mode：缺省“rgb”彩色图片返回为(x,x,3), 黑白图片返回为(x,x,1)
# -img_path: 缺省None表示不保存，否则保存在该目录下
# 返回值：图片numpy数组，标签one-hot编码后的列表
def genCaptcha(characters=string.digits+string.ascii_uppercase, capt_len=5, width=200, height=60,
               img_num=10000, color_mode='rgb', img_path=None):
    width, height, n_len, n_class = width, height, capt_len, len(characters)
    print(characters)

    images = []
    labels = []
    codeCvt = codeConvert(characters)

    # 生成一万张验证码
    for i in range(img_num):
        generator = ImageCaptcha(width=width, height=height)
        random_str = ''.join([random.choice(characters) for j in range(n_len)])
        image = generator.generate_image(random_str)

        # 不要噪点和干扰线
        # background = (255, 255, 255)
        # color = (255, 0, 0)
        # img = generator.create_captcha_image(random_str, color, background)
        # self.create_noise_dots(im, color)
        # self.create_noise_curve(im, color)
        # im = im.filter(ImageFilter.SMOOTH)

        if color_mode != "rgb":
            Grayimg = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY).astype(np.uint8)
            image = Grayimg.reshape(Grayimg.shape + (1,))
        else:
            image = np.array(image)

        if img_path != None:
            # 保存图片到指定目录：
            file_name = os.path.join(img_path, random_str+'_'+str(i)+'.jpg')
            cv2.imwrite(file_name, image)

        if i%2000 ==0:
            print(i, " captcha images are generated ...")
        images.append(image)
        labels.append(codeCvt.str2onehot(random_str))

    return np.array(images), np.array(labels)


# genCaptcha(num=1000)


if __name__ == '__main__':

    x, y = genCaptcha(string.digits+string.ascii_uppercase, img_num=5, color_mode="grayscale")
    # x, y = genCaptcha(string.digits+string.ascii_uppercase, img_num=5, color_mode="rgb", img_path="data")

    gen = data_gen.flow(x, y, batch_size=1,
        # save_to_dir=os.path.join(os.getcwd(), "data"),
        # save_prefix='F',
        # save_format='jpeg'
        )

    data = gen.next()
    # image = data[0][0]*255
    Grayimg = data[0][0]
    print(Grayimg.shape)
    # Grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # print(type(Grayimg), Grayimg.shape, Grayimg)

    cv2.imshow("GRAY", Grayimg)
    cv2.waitKey(2000)

    # retVal, image2 = cv2.threshold(Grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # print(type(image2), image2.shape)
    # cv2.imshow("DEMO", image2 / 255.0)
    # cv2.waitKey(2000)

    # 并把阈值设为0, 返回值retVal就是最优阈值
    # ret, img = cv2.threshold(Grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, img = cv2.threshold(Grayimg*255, 0, 255, cv2.THRESH_BINARY)
    # print(img)
    # cv2.imshow("DEMO", img)
    # cv2.waitKey(2000)

    # for data in gen:
    #     # image = data[0][0].astype(np.uint8)
    #     image = data[0][0]
    #     cv2.imshow("DEMO", image)
    #     cv2.waitKey(2000)

    exit(0)

    imgAug = ImageDataGenerator(rescale=1./255, rotation_range=5, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.4,)
    # # imgAug = ImageDataGenerator(rescale=1./255, shear_range=0.5)
    #
    trainGen = imageWithLable_generator(os.path.join(os.getcwd(), "../DATA/CaptchaImage"), bs=1000, mode="train", color_mode="grayscale", aug=imgAug)
    #
    # 调用测试：
    codeCvt = codeConvert(string.digits+string.ascii_uppercase)
    # 返回值为包含图片数据和标签数据的turple，每列数量为BatchSize
    for img in trainGen:
        print(len(img), len(img[0]), codeCvt.onehot2str(img[1][0]))
        cv2.imshow("GEN", img[0][0])
        cv2.waitKey(100)