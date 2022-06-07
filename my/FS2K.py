from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import parseJSON
import config
import os


# 构建数据集
class FS2K(Dataset):
    def __init__(self, selectedAttrs, transform, mode="train"):
        """
        :param selectedAttrs: []    json文件中选择的属性列表
        :param transform: 对图片进行的变换
        :param mode: str    训练模式还是验证模式
        """
        self.jsonPath = config.json_train_path if mode == "train" else config.json_test_path
        self.img_name_list, self.img_label_list = parseJSON(selectedAttrs, self.jsonPath)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        # 根据index获取对应图片名字
        img_name = self.img_name_list[index]
        # 根据index获取对应图片label
        img_label = self.img_label_list[index]
        # 获取图片，将所有图片转化为RGB格式
        img = Image.open(os.path.join(config.sketch_train_path if self.mode == "train" else config.sketch_test_path, img_name)).convert("RGB")
        if self.transform != None:
            img = self.transform(img)

        return img, img_label


if __name__ == "__main__":
    # 测试Dataset是否正常加载数据
    # 选择的属性列表
    selectedAttrs = ["hair", "gender", "earring", "smile", "frontal_face", "style"]
    # transform
    transform = transforms.Compose([
        # Resize成正方形
        transforms.Resize((128, 128)),
        # 变为tensor变量
        transforms.ToTensor(),
        # 进行标准化(标准化就是要把图片3个通道中的数据整理到[-1, 1]区间)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # 加载数据集
    data = FS2K(selectedAttrs, transform, "train")
    # 打印总共图片数量
    print(len(data))
    # 第一张图片
    print(data[0][0])
    # 第一张图片label
    print(data[0][1])
