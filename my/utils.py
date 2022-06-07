from matplotlib.pyplot import figure
import json
import matplotlib.pyplot as plt
import config


# 解析JSON文件
def parseJSON(selectedAttrs, jsonPath):
    """
    :param selectedAttrs: []    json文件中选择的属性列表
    :param jsonPath: str    存放anno_train.json或anno_test.json文件的路径
    :return: (训练集/测试集的所有素描图片名字列表，训练集/测试集素描图片所选择的属性label列表)
            example:([xxx.png,xxx.png,...],[[0,1,0,...],...])
    """
    # 读文件
    fp = open(jsonPath, 'r')
    data = json.load(fp)
    # 素描图片名字列表
    img_name_list = list()
    # 素描图片对应属性列表
    attrs_list = list()
    # 每个item是一张图片以及它的属性信息
    for img_msg in data:
        # photo1/image0001 ==> sketch1_sketch0001.png
        img_name = img_msg['image_name'].replace('photo', 'sketch').replace("/", "_").replace('image', 'sketch')
        img_name += '.png'
        img_name_list.append(img_name)
        single_img_attrs_label = list()
        for attr in selectedAttrs:
            single_img_attrs_label.append(img_msg[attr])
        attrs_list.append(single_img_attrs_label)
    return img_name_list, attrs_list


# 根据train_step_loss_record和valid_step_loss_record可视化在训练集和验证集上的损失
def loss_curve(train_step_loss_record, valid_step_loss_record):
    train_len = len(train_step_loss_record)
    x_1 = range(train_len)
    valid_len = len(valid_step_loss_record)
    x_2 = range(valid_len)
    figure(figsize=(6, 4))
    plt.plot(x_1, train_step_loss_record, c='tab:red', label='train')
    plt.plot(x_2, valid_step_loss_record, c='tab:cyan', label='valid')
    plt.xlabel('Training steps')
    plt.ylabel('Cross_entropy Loss')
    plt.legend()
    plt.show()
    # plt.savefig("./loss.png",dpi=400,bbox_inches='tight')


def acc_cur(valid_acc_average_record):
    figure(figsize=(6, 4))
    x = range(len(valid_acc_average_record))
    plt.plot(x, valid_acc_average_record, c="red")
    plt.xlabel('Steps')
    plt.ylabel('Average Acc')
    plt.legend()
    plt.show()


# 根据valid_acc_attr_record可视化在验证机上的各个属性的准确率
def acc_curve(valid_acc_attr_record, colorList):
    figure(figsize=(6, 4))
    for index, attr in enumerate(config.selectedAttrs):
        x = range(len(valid_acc_attr_record[attr]))
        color = colorList[index]
        plt.plot(x, valid_acc_attr_record[attr], c=color, label=attr)
    plt.xlabel('Steps')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    # plt.savefig("./acc.png",dpi=400,bbox_inches='tight')


if __name__ == "__main__":
    # 测试json文件是否能够正常解析
    # json文件路径
    json_anno_train_path = "/home/lab401/zsy/data/FS2K/anno_train.json"
    json_anno_test_path = "/home/lab401/zsy/data/FS2K/anno_test.json"
    # 选择的属性列表
    selectedAttrs = ['hair', "gender"]
    img_name_list, attrs_list = parseJSON(selectedAttrs, json_anno_train_path)
    print(len(img_name_list))
    print(img_name_list[0])
    print(len(attrs_list))
    print(attrs_list[0])
