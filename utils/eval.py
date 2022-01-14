import os
import sys
sys.path.append('/home/tudou/Github/tmp/MobilePose')
import torch
import numpy as np
import argparse
from dataloader import *
from coco_utils import *
from networks import *
from network import CoordRegressionNetwork
from dataset_factory import DatasetFactory



def euclidean_distance(p1, p2):
    # return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[1] - p2[1]) ** 2)
    # return np.sqrt((p1 - p2)
    p = p1 - p2
    p = p ** 2
    # b = np.sum(p, 0)

    x_square = np.sum(p, axis=1, keepdims=True)
    x_square = np.sqrt(x_square)
    x_square = np.sum(x_square)
    # x_square = x_square / len(x_square)
    return x_square


def batch_euclidean_distance(pts1, pts2):
    num = 0
    for i in range(len(pts1)):
        num += euclidean_distance(pts1[i], pts2[i])

    # for pt1, pt2 in zip(pts1, pts2):
    #     num += euclidean_distance(pt1, pt2)
    return num


def eval_coco(net, test_loader, device):
    ## generate groundtruth json
    all_coco_images_arr = [] 
    all_coco_annotations_arr = []
    num = 0
    distance = 0

    with torch.no_grad():  
        for i_batch, sample_batched in enumerate(test_loader):
            # calculate the valid loss
            num += sample_batched['image'].shape[0]
            images = sample_batched['image'].to(device)
            poses = sample_batched['pose'].numpy()
            coords, heatmaps = net(images)
            coords = coords.to("cpu").numpy()

            dist = batch_euclidean_distance(coords, poses)
            distance += dist
            num += len(images)

    print("precise: ", distance / num)



 
if __name__ == '__main__':
    
    from default import get_cfg_defaults
    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--config-file', type=str, help="config file")
    parser.add_argument('--model_path', type=str, required=True, default="")
    parser.add_argument('--result_path', type=str, required=True, default="")
    parser.add_argument('--device', type=str, required=True, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    modelpath = args.model_path

    device = torch.device(args.device)

    # user defined parameters
    # num_threads = multiprocessing.cpu_count()
    # PATH_PREFIX = "./results/{}".format(modelpath.split(".")[0])

    input_size = cfg.DATASET.INPUT_SIZE
    modelname = cfg.MODEL.BACKBONE.NAME

    test_dataset = DatasetFactory.get_test_dataset(cfg)

    print("Loading testing dataset, wait...")
    bs_test = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size= bs_test,
                            shuffle=False, num_workers = 4)

    net = CoordRegressionNetwork(n_locations=cfg.TASK.COORD_NUM, backbone=modelname).to(device)
    net.load_state_dict(torch.load(args.model_path))
    net = net.eval()
 
    eval_coco(net, test_dataloader, device)