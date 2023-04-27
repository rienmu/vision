from vit_model import vit_base_patch16_224
from de_vit import VisionTransformer

if __name__ == '__main__':
    """
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or (args.num_feature_levels > 1)
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        """
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
    from vision.models.backbone import build_backbone, build_position_encoding
    backbone = build_backbone(args)
    net = VisionTransformer(backbone=backbone, num_classes=90)
    vit = vit_base_patch16_224(num_classes=90)

    print(net.class_embed)
