




def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Segmentation Training", 
        add_help=add_help
    )

    parser.add_argument(
        "--data-path",
        default="/home/pyler/Projects/Datasets/COCO/2017/",
        type=str,
        help="dataset path"
    )
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name"
    )
    parser.add_argument(
        "--model",
        default="fcn_resnet101",
        type=str,
        help="model name"
    )
    parser.add_argument(
        "--aux-loss",
        action="store_true",
        help="auxiliar loss"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run"
    )

    parser.add_argument(
        "--workers",
        default=24,
        type=int,
        metavar="N", 
        help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "--lr",
        default=0.01, 
        type=float, 
        help="initial learning rate"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)"
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)"
    )
    parser.add_argument(
        "--lr-warmup-decay",
        default=0.01,
        type=float,
        help="the decay for lr"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        help="print frequency"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=str,
        help="path to save outputs"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path of checkpoint"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int, 
        metavar="N", 
        help="start epoch"
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", 
        action="store_true", 
        help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str, 
        help="url used to set up distributed training"
    )

    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="the weights enum name to load"
    )
    parser.add_argument(
        "--weights-backbone",
        default=None,
        type=str,
        help="the backbone weights enum name to load"
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training"
        )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
