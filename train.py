# coding: utf-8
'''
File: training.py
Project: MobilePose-PyTorch
File Created: Friday, 8th March 2019 6:53:13 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:50:27 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


# remove warning
import warnings
warnings.filterwarnings('ignore')


from network import *
from dataloader import *
from networks import *
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from dataset_factory import DatasetFactory 
import os
import multiprocessing
from tqdm import tqdm
# from yacs.config import CfgNode as CN
from default import get_cfg_defaults

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--config-file', type=str, help="config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    # user defined parameters
    num_threads = int(multiprocessing.cpu_count()/2)
    minloss = np.float("inf")
    # minloss = 0.43162785

    # gpu setting
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True

    net = CoordRegressionNetwork(n_locations=cfg.TASK.COORD_NUM, backbone=cfg.MODEL.BACKBONE.NAME).to(device)
    # net = CoordRegressionNetwork(n_locations=4, backbone=modeltype).to(device)
    net = torch.nn.DataParallel(net).to(device)

    # learning_rate = args.lr
    # batchsize = cfg.batch_size
    # inputsize = cfg.input_size
    model_name = "{}-{}-{}-{}kp".format(cfg.TASK.NAME, cfg.MODEL.BACKBONE.NAME, cfg.DATASET.INPUT_SIZE, cfg.TASK.COORD_NUM)

    # logname = modeltype+'-log.txt'
    log_name = "{}-{}-{}-{}kp.log".format(cfg.TASK.NAME, cfg.MODEL.BACKBONE.NAME, cfg.DATASET.INPUT_SIZE, cfg.TASK.COORD_NUM)

    if cfg.MODEL.PRETRAIN_PATH != "":
        # load pretrain model
        pre_net = torch.load(cfg.MODEL.PRETRAIN_PATH)
        net_dict = net.module.state_dict()
        for key in net_dict:
            if key in net_dict and net_dict[key].shape == pre_net[key].shape:
                net_dict[key] = pre_net[key]
            else:
                print("{} not exist".format(key))
        net.module.load_state_dict(net_dict)
        # net.module.load_state_dict(pre_net)
        
        for param in list(net.parameters()):
            param.requires_grad = True

    net = net.train()

    weight_dir = os.path.join(cfg.TASK.ROOT, cfg.TASK.WEIGHT_DIR)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    log_dir = os.path.join(cfg.TASK.ROOT, cfg.TASK.LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)


    train_dataset = DatasetFactory.get_train_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_BATCH_SIZE,
                            shuffle=cfg.DATALOADER.SHUFFLE, num_workers=cfg.DATALOADER.NUM_WORKERS)


    test_dataset = DatasetFactory.get_test_dataset(cfg)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
                            shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)


    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)


    train_loss_all = []
    valid_loss_all = []

    iter_num = 0

    for epoch in range(cfg.TASK.EPOCH):  # loop over the dataset multiple times
        
        train_loss_epoch = []
        train_loss_epoch_coords = []
        train_loss_epoch_hm = []

        scheduler.step()

        for i, data in enumerate(tqdm(train_dataloader)):
            # training
            images, poses = data['image'], data['pose']
            images, poses = images.to(device), poses.to(device)
            coords, heatmaps = net(images)

            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, poses)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, poses, sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)

            del data, images, poses, coords, heatmaps
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            euc_loss_value = torch.mean(euc_losses).item()
            reg_loss_value = torch.mean(reg_losses).item()
            loss_value = loss.item()
            train_loss_epoch.append(loss_value)
            train_loss_epoch_coords.append(euc_loss_value)
            train_loss_epoch_hm.append(reg_loss_value)

            # if iter_num % cfg.TASK.WRITER_FREQ == 0:
            writer.add_scalar('euc_loss', euc_loss_value, iter_num)
            writer.add_scalar('reg_loss', reg_loss_value, iter_num)
            writer.add_scalar('loss', loss_value, iter_num)

            iter_num += 1


        if epoch%2==0:

            valid_loss_epoch = []
            valid_loss_epoch_coords = []
            valid_loss_epoch_hm = []

            with torch.no_grad():  
                for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
                    # calculate the valid loss
                    images = sample_batched['image'].to(device)
                    poses = sample_batched['pose'].to(device)
                    coords, heatmaps = net(images)

                    # Per-location euclidean losses
                    euc_losses = dsntnn.euclidean_losses(coords, poses)
                    # Per-location regularization losses
                    reg_losses = dsntnn.js_reg_losses(heatmaps, poses, sigma_t=1.0)
                    # Combine losses into an overall loss
                    loss = dsntnn.average_loss(euc_losses + reg_losses)

                    del sample_batched, images, poses, coords, heatmaps

                    valid_loss_epoch.append(loss.item())
                    valid_loss_epoch_coords.append(torch.mean(euc_losses).item())
                    valid_loss_epoch_hm.append(torch.mean(reg_losses).item())

            if np.mean(np.array(valid_loss_epoch)) < minloss:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_best_file = os.path.join(weight_dir, "{}-best.pth".format(model_name))
                torch.save(net.module.state_dict(), checkpoint_best_file)
                print('==> checkpoint model saving %s'%(checkpoint_best_file))

            print('[epoch %d] train loss(coords): %.8f, train loss(hm): %.8f, train loss: %.8f,\n          valid loss(coords): %.8f, valid loss(hm): %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch_coords)), np.mean(np.array(train_loss_epoch_hm)), np.mean(np.array(train_loss_epoch)), 
                 np.mean(np.array(valid_loss_epoch_coords)), np.mean(np.array(valid_loss_epoch_hm)), np.mean(np.array(valid_loss_epoch))))

            with open(os.path.join(log_dir, log_name), 'a+') as file_output:
                file_output.write('[epoch %d] train loss(coords): %.8f, train loss(hm): %.8f, train loss: %.8f,\n          valid loss(coords): %.8f, valid loss(hm): %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch_coords)), np.mean(np.array(train_loss_epoch_hm)), np.mean(np.array(train_loss_epoch)), 
                 np.mean(np.array(valid_loss_epoch_coords)), np.mean(np.array(valid_loss_epoch_hm)), np.mean(np.array(valid_loss_epoch))))
                file_output.flush() 
        checkpoint_best_file = os.path.join(weight_dir, "{}-last.pth".format(model_name))
        torch.save(net.module.state_dict(), checkpoint_best_file)
                
    print('Finished Training')
