from ast import arg
import os
from pickletools import optimize
from random import shuffle
from statistics import mode
import numpy as np
import cv2
import argparse
import sys
import torch
from data.datasets import Cofw
from modeling import models
import logging
from engin import launch, default_argument_parser

FOD_CLASS_NAMES = ['normal', 'right_eye', 'left_eye', 'nose', 'mouth', 'chin']
CLASS_NUM = len(FOD_CLASS_NAMES)

def main(args):
    assert args.num_gpus == 1
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(args.output_dir+'log.txt',"w", encoding="UTF-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    dataset = Cofw(proj_dir=args.proj_dir, data_dir=args.data_dir, batch_size=args.batch_size,
                            input_size=args.input_size, class_num=CLASS_NUM,
                            fine_tune=args.fine_tune)
    if args.eval_only:
        net = models.FODNet(class_num = CLASS_NUM,
                    input_size = args.input_size, 
                    fine_tune=args.fine_tune)
        state_dict = torch.load(args.model_weights,map_location=torch.device('cuda'))
        net.load_state_dict(state_dict)
        net.to('cuda')
        logger.info('start evaluating val dataset...')
        net.eval()
        iters = int(dataset.val_num()/args.batch_size)
        total = iters*args.batch_size
        dataloader = dataset.data_generator(label_file_name='val.txt',shuffle=False)
        predictions = []
        correct = 0
        for it in range(iters):
            if it%10==0:
                logger.info('--> iter {}/{}'.format(it+1,iters))
            fod_x, fod_label = next(dataloader)
            fod_x = fod_x.to('cuda')
            pred = net(fod_x).detach().cpu()
            for i, fod_real in enumerate(fod_label):
                fod_real = fod_real.tolist()
                one_num = fod_real.count(1)
                fod_pred_idxs = [j for j,x in enumerate(pred[i].tolist()) if x > args.thresholds[j]]
                fod_real_idxs = [i for i,x in enumerate(fod_real) if x == 1]
                # logger.info(fod_pred_idxs)
                # logger.info(fod_real_idxs)
                if fod_real_idxs == fod_pred_idxs:
                    correct += 1
            predictions.append(pred)
        
        logger.info("fod val ==> correct:{}, total:{}, correct_rate:{}".format(correct, total, 1.0 * correct / total))
        # -------------------------------------------------------
        iters = int(dataset.test_num()/args.batch_size)
        total = iters*args.batch_size
        dataloader = dataset.data_generator(label_file_name='test.txt',shuffle=False)
        predictions = []
        correct = 0
        for it in range(iters):
            if it%10==0:
                logger.info('--> iter {}/{}'.format(it+1,iters))
            fod_x, fod_label = next(dataloader)
            fod_x = fod_x.to('cuda')
            pred = net(fod_x).detach().cpu()
            for i, fod_real in enumerate(fod_label):
                fod_real = fod_real.tolist()
                one_num = fod_real.count(1)
                fod_pred_idxs = [j for j,x in enumerate(pred[i].tolist()) if x > args.thresholds[j]]
                fod_real_idxs = [i for i,x in enumerate(fod_real) if x == 1]
                # logger.info(fod_pred_idxs)
                # logger.info(fod_real_idxs)
                if fod_real_idxs == fod_pred_idxs:
                    correct += 1
            predictions.append(pred)
        
        logger.info("fod test ==> correct:{}, total:{}, correct_rate:{}".format(correct, total, 1.0 * correct / total))
        logger.info('evaluation finished!')
        return
        
    net = models.FODNet(class_num = CLASS_NUM,
                        input_size = args.input_size, 
                        fine_tune=args.fine_tune,
                        fine_tune_model_file=args.model_weights)
    net.to('cuda')
    optimizer = torch.optim.AdamW(net.parameters())
    logger.info(net)
    logger.info('start training...')
    net.train()
    dataloader = dataset.data_generator(label_file_name='train.txt',shuffle=True)
    for epoch in range(args.epochs):
        logger.info('epoch {}/{}'.format(epoch+1,args.epochs))
        iters = int(dataset.train_num()/args.batch_size)
        for it in range(iters):
            fod_x, fod_label = next(dataloader)
            fod_x = fod_x.to('cuda')
            fod_label = fod_label.to('cuda')
            pred, loss_dict = net(fod_x,gt_label=fod_label)
            losses = sum(loss_dict.values())
            logger.info('--> iter {}/{},  total loss: {}'.format(it+1,iters,losses))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    torch.save(net.state_dict(),args.output_dir+'/model_final.pth')
    logger.info('training finished!')
        
if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    # args.eval_only = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

