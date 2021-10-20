from models.sseg.deeplabv3plus import DeepLabV3Plus
from models.sseg.fcn import FCN
from models.sseg.pspnet import PSPNet
from models.sseg.unet import UNet


def get_model(model, backbone, pretrained, nclass, lightweight):
    if model == "fcn":
        model = FCN(backbone, pretrained, nclass, lightweight)
    elif model == "pspnet":
        model = PSPNet(backbone, pretrained, nclass, lightweight)
    elif model == "deeplabv3plus":
        model = DeepLabV3Plus(backbone, pretrained, nclass, lightweight)
    elif model == "unet":
        model = UNet(backbone, pretrained, nclass, lightweight)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    params_num = sum(p.numel() for p in model.parameters())
    # 计算模型的参数量 以百万为单位
    print("\nParams: %.1fM" % (params_num / 1e6))

    return model
