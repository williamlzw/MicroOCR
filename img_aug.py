import cv2
import abc
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import math


def cv2pil(image):
    """
    将bgr格式的numpy的图像转换为pil
    :param image:   图像数组
    :return:    Image对象
    """
    assert isinstance(image, np.ndarray), 'input image type is not cv2'
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_pil_image(image):
    """
    将图像统一转换为PIL格式
    :param image:   图像
    :return:    Image格式的图像
    """
    if isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return image
    elif isinstance(image, np.ndarray):
        return cv2pil(image)


def get_cv_image(image):
    """
    将图像转换为numpy格式的数据
    :param image:   图像
    :return:    ndarray格式的图像数据
    """
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return pil2cv(image)


def pil2cv(image):
    """
    将Image对象转换为ndarray格式图像
    :param image:   图像对象
    :return:    ndarray图像数组
    """
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


class TransBase(object):
    """
    数据增广的基类
    """

    def __init__(self, probability=1.):
        """
        初始化对象
        :param probability:     执行概率
        """
        super(TransBase, self).__init__()
        self.probability = probability

    @abc.abstractmethod
    def trans_function(self, _image):
        """
        初始化执行函数，需要进行重载
        :param _image:  待处理图像
        :return:    执行后的Image对象
        """
        pass

    # @utils.zlog
    def process(self, _image):
        """
        调用执行函数
        :param _image:  待处理图像
        :return:    执行后的Image对象
        """
        if np.random.random() < self.probability:
            return self.trans_function(_image)
        else:
            return _image

    def __call__(self, _image):
        """
        重载()，方便直接进行调用
        :param _image:  待处理图像
        :return:    执行后的Image
        """
        return self.process(_image)



class SightTransfer(TransBase):
    """
    随机视角变换
    """
    
    def setparam(self):
        self.horizontal_sight_directions = ('left', 'mid', 'right')
        self.vertical_sight_directions = ('up', 'mid', 'down')
        self.angle_left_right = 5
        self.angle_up_down = 5
        self.angle_vertical = 5
        self.angle_horizontal = 5

    def trans_function(self, image):
        horizontal_sight_direction = self.horizontal_sight_directions[random.randint(0, 2)]
        vertical_sight_direction = self.vertical_sight_directions[random.randint(0, 2)]
        image = get_cv_image(image)
        image = self.sight_transfer([image], horizontal_sight_direction, vertical_sight_direction)
        image = image[0]
        image = get_pil_image(image)
        return image
    
    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

    def left_right_transfer(self, img, is_left=True, angle=None):
        """ 左右视角，默认左视角
        :param img: 正面视角原始图片
        :param is_left: 是否左视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.angle_left_right  # self.rand_reduce(self.angle_left_right)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_left:
            pts2 = np.float32([[0, 0], [0, size_src[1]],
                               [size_src[0], interval], [size_src[0], size_src[1] - interval]])
        else:
            pts2 = np.float32([[0, interval], [0, size_src[1] - interval],
                               [size_src[0], 0], [size_src[0], size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def up_down_transfer(self, img, is_down=True, angle=None):
        """ 上下视角，默认下视角
        :param img: 正面视角原始图片
        :param is_down: 是否下视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.rand_reduce(self.angle_up_down)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_down:
            pts2 = np.float32([[interval, 0], [0, size_src[1]],
                               [size_src[0] - interval, 0], [size_src[0], size_src[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_src[1]],
                               [size_src[0], 0], [size_src[0] - interval, size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def vertical_tilt_transfer(self, img, is_left_high=True):
        """ 添加按照指定角度进行垂直倾斜(上倾斜或下倾斜，最大倾斜角度self.angle_vertical一半）
        :param img: 输入图像的numpy
        :param is_left_high: 图片投影的倾斜角度，左边是否相对右边高
        """
        angle = self.rand_reduce(self.angle_vertical)
    
        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
    
        # 计算图片进行上下倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
        size_target = (int(math.cos((float(angle) / 180) * math.pi) * shape[1]), shape[0] + interval)
        # 目标图像上四个顶点的坐标
        if is_left_high:
            pts2 = np.float32([[0, 0], [0, size_target[1] - interval],
                               [size_target[0], interval], [size_target[0], size_target[1]]])
        else:
            pts2 = np.float32([[0, interval], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0], size_target[1] - interval]])
    
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def horizontal_tilt_transfer(self, img, is_right_tilt=True):
        """ 添加按照指定角度进行水平倾斜(右倾斜或左倾斜，最大倾斜角度self.angle_horizontal一半）
        :param img: 输入图像的numpy
        :param is_right_tilt: 图片投影的倾斜方向（右倾，左倾）
        """
        angle = self.rand_reduce(self.angle_horizontal)
            
        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        
        # 计算图片进行左右倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        size_target = (shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_right_tilt:
            pts2 = np.float32([[interval, 0], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0] - interval, size_target[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_target[1]],
                               [size_target[0] - interval, 0], [size_target[0], size_target[1]]])
        
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def sight_transfer(self, images, horizontal_sight_direction, vertical_sight_direction):
        """ 对图片进行视角变换
        :param images: 图片列表
        :param horizontal_sight_direction: 水平视角变换方向
        :param vertical_sight_direction: 垂直视角变换方向
        :return:
        """
        flag = 0
        img_num = len(images)
        # 左右视角
        if horizontal_sight_direction == 'left':
            flag += 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif horizontal_sight_direction == 'right':
            flag -= 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass
        # 上下视角
        if vertical_sight_direction == 'down':
            flag += 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif vertical_sight_direction == 'up':
            flag -= 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass
        
        # 左下视角 或 右上视角
        if abs(flag) == 2:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
                
            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        # 左上视角 或 右下视角
        elif abs(flag) == 1:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)

            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass
        
        return images


class Blur(TransBase):
    """
    随机高斯模糊
    """

    def setparam(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_pil_image(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        return image


class MotionBlur(TransBase):
    """
    随机运动模糊
    """

    def setparam(self, degree=5, angle=180):
        self.degree = degree
        self.angle = angle

    def trans_function(self, image):
        image = get_pil_image(image)
        angle = random.randint(0, self.angle)
        M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        image = image.filter(ImageFilter.Kernel(size=(self.degree, self.degree), kernel=motion_blur_kernel.reshape(-1)))
        return image

class RandomHsv(TransBase):
    def setparam(self, hue_keep=0.1, saturation_keep=0.7, value_keep=0.4):
        self.hue_keep = hue_keep
        self.saturation_keep = saturation_keep
        self.value_keep = value_keep
    
    def trans_function(self, image):
        image = get_cv_image(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 色调，饱和度，亮度
        hsv[:, :, 0] = hsv[:, :, 0] * (self.hue_keep + np.random.random() * (1 - self.hue_keep))
        hsv[:, :, 1] = hsv[:, :, 1] * (self.saturation_keep + np.random.random() * (1 - self.saturation_keep))
        hsv[:, :, 2] = hsv[:, :, 2] * (self.value_keep + np.random.random() * (1 - self.value_keep))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image = get_pil_image(image)
        return image

class Smudge(TransBase):
    def setparam(self):
        pass

    def trans_function(self, image):
        image = get_cv_image(image)
        smu = cv2.imread("smu.jpg")
        rows = self.rand_reduce(smu.shape[0] - image.shape[0])
        cols = self.rand_reduce(smu.shape[1] - image.shape[1])
        add_smu = smu[rows:rows + image.shape[0], cols:cols + image.shape[1]]
        image = cv2.bitwise_not(image)
        image = cv2.bitwise_and(add_smu, image)
        image = cv2.bitwise_not(image)
        image = get_pil_image(image)
        return image

    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

class DataProcess:
    def __init__(self):
        """
        文本数据增广类
        """
        self.sight_transfer = SightTransfer(probability=0.5)
        self.blur = Blur(probability=0.3)
        self.motion_blur = MotionBlur(probability=0.3)
        self.rand_hsv = RandomHsv(probability=0.3)
        self.sight_transfer.setparam()
        self.blur.setparam()
        self.motion_blur.setparam()
        self.rand_hsv.setparam()


    def aug_img(self, img):
        img = self.motion_blur.process(img)
        img = self.blur.process(img)
        img = self.sight_transfer.process(img)
        img = self.rand_hsv.process(img)
        return img


if __name__ == '__main__':
    pass
    img = Image.open('002.png')
    aa = Smudge()
    aa.setparam()
    img = aa.trans_function(img)
    img.save('00001.jpg')
    # data_augment = DataAug()
    # augmented_img = data_augment.aug_img(img)
    # augmented_img.show()