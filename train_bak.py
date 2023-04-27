# -*- coding: utf-8 -*-

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchvision import transforms, datasets
from tqdm import tqdm

from vision.models import resnet,deformable_vit
from vision.models.vit_model import vit_base_patch16_224_in21k as create_model
from create_dataset import AlbumentationsDataset,strong_aug,val_aug


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # data_transform = {
    #     "train": strong_aug(0.5),
    #     'val': val_aug(0.5)
    # }
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = r"F:\__Earsenal\images"
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "valid")
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = AlbumentationsDataset(root=train_root,
                                          transform=data_transform["train"])
    train_dataset = datasets.ImageFolder(root=train_root,transform=data_transform['train'])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w',encoding='utf-8') as json_file:
        json_file.write(json_str)

    batch_size = 24
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=True)

    validate_dataset = AlbumentationsDataset(root=val_root,
                                             transform=data_transform["val"])
    validate_dataset = datasets.ImageFolder(root=val_root,transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, pin_memory=True)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    #net = deformable_vit.DeformableViT(num_classes=90)
    # net = torchvision.models.vit_b_16(weights=None)
    # net.heads = nn.Linear(768,90)
    # net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    # net.classifier = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, 90),
    # )
    # net.classifier= nn.Sequential(
    #         nn.Linear(512 * 7 * 7, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(4096, 90),
    #     )
    #net = torchvision.models.convnext_base(num_classes=90,weights=None)
    #net.head = nn.Linear(1024, 90)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    net = resnet.resnet50()
    #model_weight_path = r"../vision_transformer/vit_base_patch16_224_in21k.pth"
    #assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    #model = torch.load(model_weight_path).to('cpu')


    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    #resnet
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 90)


    #vit
    #net.head = nn.Linear(768,90)



    net.to(device)
    print(net)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)

    epochs = 200
    start_epoch = 0
    best_acc = 0.0
    save_path = r'F:\_checkpoint\convnext_no'
    train_steps = len(train_loader)
    test_only = False
    for epoch in range(start_epoch,epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        if not test_only:
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                #print(predict_y)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        #if val_accurate > best_acc:
         #   best_acc = val_accurate
        with open(os.path.join(save_path,"log.txt"),"a") as f:
            f.write("epoch:{:d},acc:{:.4f},loss:{:.4f},lr:{:.4f}\n".format(epoch,val_accurate,running_loss / train_steps,optimizer.defaults['lr']))
        torch.save(net, os.path.join(save_path,"convnext_no-{:d}.pth".format(epoch)))

    print('Finished Training')
    f.close()

if __name__ == '__main__':
    main()
