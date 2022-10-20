import argparse
import logging
import os
import sys

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--input-size", type=int, default=96, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size.")
    parser.add_argument("--fine-tune", type=bool, default=False, help="Fine tune based on Vgg16.")
    parser.add_argument("--model-weights", type=str, default="output/model_final.pth", help="pretrained model weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Train epochs.")
    parser.add_argument("--thresholds", type=list, default=[0.5,0.5,0.5,0.5,0.5,0.5], help="thresholds for evaluation.")
    parser.add_argument("--proj-dir", type=str, default="./", help="Project directory.")
    parser.add_argument("--data-dir", type=str, default="datasets/cofw/", help="data directory.")
    parser.add_argument("--output-dir", type=str, default="output/", help="output directory.")
    return parser
