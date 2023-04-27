import os
import json
import time
from collections import Counter

import numpy
from torch import nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from vision.models.resnet import resnet50
from vision.models.de_vit import VisionTransformer
import xlwt


def main(args):
    # 创建excel
    book = xlwt.Workbook()
    every_animal = book.add_sheet('每种动物准确率')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    count = 0
    supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]  # 支持的文件后缀类型
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # logstr
    logstr = ''
    timer = time.ctime()
    timer = timer.replace(' ', '-')
    timer = timer.replace(':', '-')
    # create model
    from vision.models.backbone import build_backbone, build_position_encoding

    backbone = build_backbone(args)
    net = VisionTransformer(backbone=backbone, num_classes=90)
    # load model weights
    model_weight_path = r"F:\_checkpoint\de_vit_no/resnet500.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = torch.load(model_weight_path)
    model.eval()
    count_a = 0
    index_list = []
    cr_list = []

    for name in class_indict:
        index_list.clear()
        img_path = r"F:\__Earsenal\images\test/"
        class_name = class_indict[str(name)]
        img_path = img_path + class_name + '/'
        images = os.listdir(img_path)
        images = [os.path.join(img_path, i) for i in os.listdir(img_path)
                  if os.path.splitext(i)[-1] in supported]
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        index_label = 0
        for i in images:
            count += 1
            img = Image.open(i)
            img = img.convert('RGB')
            # plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            plt.title(print_res)
            index = numpy.argmax(predict.numpy())
            index_list.append(index)
            index_label += 1
            strs = "{:} class: {:10}   prob: {:.3} index: {:}".format(index_label, class_indict[str(index)],
                                                                      predict[index].numpy(), index)
            # print(strs)
            # excel文件写入
            # 日志文本
            logstr = logstr + strs + '\n'

            # for i in range(len(predict)):
            # print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
            # predict[i].numpy()))
            # print('\n')
            # plt.show()
        cor_rate = Counter(index_list)
        print(count_a, cor_rate)
        cr = cor_rate[count_a] / len(index_list)
        cr_list.append(cr)
        count_a += 1
        logstr = logstr + str(cor_rate) + '\n'
        logstr = logstr + '{:} 正确率:{:.3}\n'.format(class_indict[str(count_a - 1)], cr)

        # excel写入
        every_animal.write(count_a - 1, 0, class_indict[str(count_a - 1)])
        every_animal.write(count_a - 1, 1, cr)

        if not os.path.exists('./log-{:}-{:}.txt'.format(timer, str(len(class_indict)))):
            open('./log-{:}-{:}.txt'.format(timer, str(len(class_indict))), 'x')
        with open('./log-{:}-{:}.txt'.format(timer, str(len(class_indict))), 'w') as file_a:
            file_a.write(logstr)
            file_a.close()
    logstr = logstr + str(numpy.average(cr_list))
    file = open('./log-{:}-{:}.txt'.format(timer, str(len(class_indict))), 'w')
    file.write(logstr)
    file.close()
    book.save('{:}-{:}.xls'.format(timer, len(class_indict)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--lr_backbone', default='1e-5', type=float)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--num_feature_levels', default='4', type=int)
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    args = parser.parse_args()

    main(args)
