# -*- coding: utf-8 -*-

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from vision.models import resnet, de_vit
from vision.models.vit_model import vit_base_patch16_224_in21k as create_model
from create_dataset import AlbumentationsDataset, strong_aug, val_aug


def main(args):
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

    data_root = args.data_root
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "valid")
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    train_dataset = AlbumentationsDataset(root=train_root,
                                          transform=data_transform["train"])
    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform['train'])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=True)

    validate_dataset = AlbumentationsDataset(root=val_root,
                                             transform=data_transform["val"])
    validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, pin_memory=True)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    from vision.models.backbone import build_backbone, build_position_encoding

    backbone = build_backbone(args)
    net = de_vit.VisionTransformer(backbone=backbone, num_classes=1000)
    #net = torch.load(r"F:\_checkpoint\de_vit_no\resnet50111.pth")
    net.to(device)
    print(net)
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # print model's params msg
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    for n, p in net.named_parameters():
        print(n)
    params = [p for p in net.parameters() if p.requires_grad]
    # construct an optimizer
    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in net.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
    #                                                                                                args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr,
    #     },
    #     {
    #         "params": [p for n, p in net.named_parameters() if
    #                    match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     }
    #     # },
    #     # {
    #     #     "params": [p for n, p in net.named_parameters() if
    #     #                match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #     #     "lr": args.lr * args.lr_linear_proj_mult,
    #     # }
    # ]
    if args.sgd:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint


    best_acc = 0.0
    save_path = args.save_path
    assert os.path.exists(save_path), 'save path {:} does not exists'.format(save_path)

    train_steps = len(train_loader)
    test_only = False
    for epoch in range(start_epoch, args.epochs):
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
                                                                         args.epochs,
                                                                         loss)
        lr_scheduler.step()
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
                # print(predict_y)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # if val_accurate > best_acc:
        #   best_acc = val_accurate
        with open(os.path.join(save_path, "log.txt"), "a") as f:
            f.write("epoch:{:d},acc:{:.4f},loss:{:.4f},lr:{:.4}\n".format(epoch, val_accurate, running_loss / train_steps, optimizer.defaults['lr']))
        if args.resume:
            # net.load_state_dict(checkpoint['model'], strict=False)
            # start_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint['model'] = net.state_dict()

        torch.save(net, os.path.join(save_path, "resnet50{:d}.pth".format(epoch)))

    print('Finished Training')
    f.close()


if __name__ == '__main__':

    if __name__ == "__main__":
        """
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or (args.num_feature_levels > 1)
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        """
        import argparse

        parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--lr', default='0.005', type=float)
        parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str,
                            nargs='+')
        parser.add_argument('--lr_drop', default=10, type=int)
        parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
        parser.add_argument('--sgd',action='store_true')
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--position_embedding', default='sine', type=str)
        parser.add_argument('--data_root',default=r"F:\__Earsenal\iamgenet_1k",type=str)
        parser.add_argument('--batch_size',default=32,type=int)
        parser.add_argument('--save_path',default=r'F:\_checkpoint\de_vit_imagenet_1k',type=str)
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--num_feature_levels', default='4', type=int)
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--resume', default='', type=str)

        # backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--lr_backbone', default='0.001', type=float)
        parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')


        args = parser.parse_args()
        CUDA_LAUNCH_BLOCKING = 1
        main(args)
