import argparse
from configs.env_config import *

splits = {
    # 'aims_sep_10percent': {'test': 'aims_split_2023_test.json', 'val': 'aims_split_2023_val.json'},
    'aims_sep': {'test': 'aims_split_2023_test.json', 'val': 'aims_split_2023_val.json'},
    'aims_oct': {'test': 'instances_default.json'},
    'kaggle': {'test': 'mmdet_split_test.json', 'val': 'mmdet_split_val.json'}
}

roots = {
    'aims_sep': DATA_DIRECTORY_TARGET,
    'aims_oct': DATA_DIRECTORY_TARGET2,
    'kaggle': DATA_DIRECTORY_SOURCE
}

def get_arguments():
    parser = argparse.ArgumentParser(
                    prog = 'Evaluation',
                    description = 'Evaluates the model on the given dataset with the specified split.')

    parser.add_argument('--dataset', default='aims_sep', choices=list(splits.keys()))
    parser.add_argument('--model', default='yolov5')
    parser.add_argument('--enhancement', default=None, choices=['relightnet'])
    parser.add_argument('--online', default=False)
    parser.add_argument('--ckpt', default=None, help="Checkpoint, like GoProv3_AIMS_23_resplit-wbbox-backbone-adaptivemomentumhighafter1k-0.1scoreteachonline-nolightnetonteacher-gkernimmask-copyteachertostudentevery1k-randomresize_teacher_1000.pth")
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--subset', default=-1, help="Evaluation subset size, -1 for no subset.")
    return parser.parse_args()