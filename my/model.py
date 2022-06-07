from torchvision import models
from torch import nn
import torch
import config


# 自定义卷积层提取特征 输入图片大小必须为[3, 128, 128]
class BaseFeatureExtraction(nn.Module):
    def __init__(self):
        super(BaseFeatureExtraction, self).__init__()

        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # 卷积层
        # input image size: [3, 128, 128]
        # output image size: [256, 8, 8]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # 64 * 126 * 126
            # 卷积层之后总会添加BatchNorm2d进行数据的标准化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # 64 * 63 * 63

            nn.Conv2d(64, 128, 3, 1, 1),
            # 128 * 61 * 61
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # 128 * 31 * 31

            nn.Conv2d(128, 256, 3, 1, 1),
            # 256 * 29 * 29
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
            # 256 * 8 * 8
        )

    def forward(self, image):
        return self.cnn_layers(image)


# 通过预训练模型提取特征
class FeatureExtraction(nn.Module):
    def __init__(self, isPretrained, modelName="Resnet18"):
        """
        :param isPretrained: boolean    模型是否经过预训练
        :param modelName: str    卷积采用的预训练模型名字
        """
        super(FeatureExtraction, self).__init__()

        if modelName == "Resnet18":
            # 创建模型
            self.model = models.resnet18(pretrained=isPretrained)
            self.model.load_state_dict(torch.load(config.resnet18))
        elif modelName == "Resnet34":
            self.model = models.resnet34(pretrained=isPretrained)
            self.model.load_state_dict(torch.load(config.resnet34))
        elif modelName == "Resnet50":
            self.model = models.resnet50(pretrained=isPretrained)
            self.model.load_state_dict(torch.load(config.resnet50))
        elif modelName == "AlexNet":
            self.model = models.alexnet(pretrained=isPretrained)
            self.model.load_state_dict(torch.load(config.AlexNet))
        elif modelName == "VGG16":
            self.model = models.vgg16(pretrained=isPretrained)
            self.model.load_state_dict(torch.load(config.VGG16))

    def forward(self, image):
        return self.model(image)


# 自定义的全连接层
class FullConnect(nn.Module):
    # 自定义卷积层:input_dim = 256 * 8 * 8
    # 预训练模型Resnet18: input_dim = 512
    # 预训练模型VGG16: input_dim = 25088
    # 预训练模型AlexNet: input_dim = 9216
    def __init__(self, input_dim=256 * 8 * 8, output_dim=2):
        """
        :param input_dim: 全连接层输入维度（卷积层输出维度）
        :param output_dim: 全连接层输出维度（分类类别数目）
        """
        super(FullConnect, self).__init__()

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, image):
        # 从1维展开（将每一张[C,W,H]图片变成[C*W*H]）
        image = image.flatten(1)
        # 经过全连接层处理
        image = self.fc_layers(image)
        return image


# 多属性分类模型
# 思路：（1）首先将图片都通过卷积层提取特征
#      （2）不同的属性通过不同的全连接层预测其label
class MultiAttrsClassifier(nn.Module):
    def __init__(self, modelName="BaseModel", isPretrained=True):
        """
        :param modelName: str    卷积采用的模型名字("BaseModel", "Resnet18", "VGG16", "AlexNet")
        :param isPretrained: boolean    模型是否经过预训练
        """
        super(MultiAttrsClassifier, self).__init__()
        # 初始化卷积模型，不同卷积模型最后的输出维度不同
        if modelName == "BaseModel":
            self.cnn_out_feature_dim = 256 * 8 * 8
            self.featureExtractor = BaseFeatureExtraction()
        elif modelName == "Resnet18":
            # self.cnn_out_feature_dim = 512
            self.cnn_out_feature_dim = 1000
            self.featureExtractor = FeatureExtraction(isPretrained, modelName)
        elif modelName == "Resnet34":
            self.cnn_out_feature_dim = 1000
            self.featureExtractor = FeatureExtraction(isPretrained, modelName)
        elif modelName == "Resnet50":
            self.cnn_out_feature_dim = 1000
            self.featureExtractor = FeatureExtraction(isPretrained, modelName)
        elif modelName == "VGG16":
            self.cnn_out_feature_dim = 1000
            self.featureExtractor = FeatureExtraction(isPretrained, modelName)
        elif modelName == "AlexNet":
            self.cnn_out_feature_dim = 1000
            self.featureExtractor = FeatureExtraction(isPretrained, modelName)

        # 不同属性使用不同的全连接层
        self.FC_hair = FullConnect(input_dim=self.cnn_out_feature_dim)
        self.FC_gender = FullConnect(input_dim=self.cnn_out_feature_dim)
        self.FC_earring = FullConnect(input_dim=self.cnn_out_feature_dim)
        self.FC_smile = FullConnect(input_dim=self.cnn_out_feature_dim)
        self.FC_frontal = FullConnect(input_dim=self.cnn_out_feature_dim)
        self.FC_style = FullConnect(input_dim=self.cnn_out_feature_dim, output_dim=3)

    def forward(self, image):
        # 得到图像特征
        features = self.featureExtractor(image)
        # 不同fc对不同属性进行分类
        # 修改
        hair = self.FC_hair(features)
        gender = self.FC_gender(features)
        earring = self.FC_earring(features)
        smile = self.FC_smile(features)
        frontal = self.FC_frontal(features)
        style = self.FC_style(features)

        # 修改
        return hair, gender, earring, smile, frontal, style
        # return style


if __name__ == "__main__":
    # model = BaseFeatureExtraction()
    model = FeatureExtraction(False, "VGG16")
    # model = MultiAttrsClassifier(False)
    print(model)
