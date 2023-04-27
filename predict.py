import os
import json

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         # mean=(0.4558,0.4558,0.4558),std=(0.2741,0.2741,0.2741)
         transforms.Normalize([0.4558, 0.4558, 0.4558], [0.2741, 0.2741, 0.2741])])

    # load image
    img_path = r"F:\__Earsenal\images\train\bee/1.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = img.convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]

    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = my_models.ResNet(num_classes=90).to(device)
    from vision.models import resnet
    model = resnet.resnet50()
    model.fc = nn.Linear(2048, 90)
    model.to(device)
    # load model weights
    model_weight_path = r"F:\_checkpoint\resnext/resnet500.pth"
    model = torch.load(model_weight_path)
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(img.to(device))
        #predict_y = torch.topk(output, 5, dim=1)[1]
        # loss = loss_function(outputs, test_labels)
        # predict_y = torch.max(outputs, dim=1)[1]
        output = torch.squeeze(model(img.to(device))).cpu()
        print(output)
        _, predic = output.topk(5, 0, True, True)
        predic = predic.t()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict, dim=0).numpy()
        # predict = torch.topk(predict,5)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)

    for i in predic:
        i = i.numpy()
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
