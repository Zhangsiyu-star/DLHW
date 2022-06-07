from torch.utils.data import DataLoader
from torchvision import transforms
from FS2K import FS2K
import torch

def collate_fn(batch):
    """
    :param batch: 根据dataset取出前batch_size个数据，然后弄成一个列表
    :return: real_batch
    """
    # zip(*batch) --> [(img1(tensor),img2(tensor),...),(label1(list),label2(list),...)]
    images, labels = tuple(zip(*batch))
    # 将images的每个item转为tensor
    # Tensor(img1(tensor),img2(tensor),...)
    images = torch.stack(images, dim=0)
    # 将labels的每个item转为tensor
    # Tensor(label1(tensor),label2(tensor),...)
    labels = torch.as_tensor(labels)
    return images, labels


# DataLoader
def get_loader(batch_size, selectedAttrs, mode='train', transform=None):
    dataset = FS2K(selectedAttrs, transform, mode)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), drop_last=True, collate_fn=collate_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), drop_last=True)

    return dataloader


# 测试DataLoader是否正常
# selectedAttrs = ["hair", "gender", "earring", "smile", "frontal_face", "style"]
# transform = transforms.Compose([
#     # Resize成正方形
#     transforms.Resize((128, 128)),
#     # 变为tensor变量
#     transforms.ToTensor(),
#     # 进行标准化(标准化就是要把图片3个通道中的数据整理到[-1, 1]区间)
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# data = get_loader(4, selectedAttrs, "train", transform)
# for batch_data in data:
#     imgs, labels = batch_data
#     print(imgs[0].shape)
#     print(labels[0])
#     break
