import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('GaoFen - Change Detection')
        # parser.add_argument("--data-root", type=str,
        #                     default="/home/kelin/data/SenseEarth2020-ChangeDetection")
        parser.add_argument("--exp_name", type=str,
                            default="fcn_resnet34_lr1e-4")
        parser.add_argument("--train_root", type=str,
                            default="/home/kelin/data")
        parser.add_argument("--val_root", type=str,
                            default="/home/kelin/data")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--val_batch_size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--model", type=str, default="fcn")
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--pretrain_from", type=str)
        parser.add_argument(
            "--pretrained", dest="pretrained", action="store_true")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
