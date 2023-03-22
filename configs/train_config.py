import argparse
from .env_config import *

MODEL = 'yolo'

LIGHTNET = "..\DANNet\snapshots\yolo\scintillating-cake-209_light_latest.pth"
DARKNET = "..\DANNet\snapshots/yolo/dual-discrim-epoch200-yang_dark_5000.pth"
D_DS = "..\DANNet\snapshots\yolo\dual-discrim-epoch200-yang_d_ds_5000.pth"
D_ADV = "..\DANNet\snapshots\yolo\dual-discrim-epoch200-yang_d_adv_5000.pth"

BATCH_SIZE = 2
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
LEARNING_RATE_YOLO = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
MOMENTUM_TEACHER = 0.6 # 0.9995
TEMPERATURE_TEACHER = 0.04
TEACHER_SCORE_THRESH = 0.6
QUEUE_SIZE = 200

NUM_STEPS = 1
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = './snapshots/'+MODEL
STD = 0.05

# enhancement weights
e_wgts = dict(
    tv=10,
    ssim=1,
    expz=1,
    ciconv=1.0
)

# fixed from yang_model
yolo_loss_weights = dict(
    loss_cls=0.22499999999999998,
    loss_bbox=0.037500000000000006,
    loss_obj=2.0999999999999996
)

yolov8_loss_weights = dict(
    loss_cls=0.5, # was 0.5
    loss_bbox=0.1, # was 7.5
    loss_dfl=1.5/4
)

fasterrcnn_loss_weights = dict(
    loss_rpn_cls=1.0,
    loss_rpn_bbox=1.0,
    loss_cls=1.0,
    loss_bbox=1.0,
)

def get_arguments():
    parser = argparse.ArgumentParser(description="Underwater Domain Adaptation")
    parser.add_argument('--model', default='yolov5', choices=['yolov5', 'yolov8', 'fasterrcnn'])
    parser.add_argument('--run-name', type=str)

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--queue-size", type=int, default=QUEUE_SIZE,
                        help="Size of object injection queue.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--learning-rate-yolo", type=float, default=LEARNING_RATE_YOLO,
                        help="Base learning rate for yolo.")
    parser.add_argument("--teacher-score-thresh", type=float, default=TEACHER_SCORE_THRESH,
                        help="Teacher pseudolabelling score threshold")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps/batches.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--lightnet", type=str, default=LIGHTNET,
                        help="Lightnet checkpoint.")
    parser.add_argument("--darknet", type=str, default=DARKNET,
                        help="Darknet checkpoint.")
    parser.add_argument("--train-style", type=bool, default=False,
                        help="Whether to train discriminators/relighting networks.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots.")
    return parser.parse_args()
