import heapq
import os

import cv2
import numpy as np
import torch
import torchvision


class FODNet(torch.nn.Module):
    def __init__(self, class_num, input_size, fine_tune=True, fine_tune_model_file='imagenet'):
        super().__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.fine_tune_model_file = fine_tune_model_file
        self.fine_tune = fine_tune
        self.stride = 4
        if fine_tune:
            self.stride = 32
            self.model = self.fine_tune_model()
        else:
            self.model = self._create_model()
            
    def build_backbone(self):
        backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
        )
        
        return backbone
    
    def build_neck(self):
        neck = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1,end_dim=-1),
            torch.nn.Linear(in_features=int((self.input_size/self.stride)**2*64),out_features=128),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=128,out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
        )
        return neck
    
    def _create_model(self):
        self.model = torch.nn.Sequential(
            self.build_backbone(),
            self.build_neck(),
            self.build_classification_head(),
        )

        return self.model

    def build_classification_head(self):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=256,out_features=self.class_num),
            torch.nn.Sigmoid(),
        )

    def fine_tune_model(self):
        if self.fine_tune_model_file != '':
            vgg16 = torchvision.models.vgg16(pretrained=False)
            vgg16.load_state_dict(torch.load(self.fine_tune_model_file))
            del vgg16.classifier
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
            del vgg16.classifier
        self.model = torch.nn.Sequential(
            vgg16,
            self.build_neck(),
            self.build_classification_head(),
        )

        return self.model

    def inference(self,x):
        return self.model(x)

    def forward(self, x, gt_label = None):
        if not self.training:
            return self.inference(x)
        output = self.model(x)
        loss_dict = {'total_loss':torch.nn.functional.binary_cross_entropy(output,gt_label)}
        return output, loss_dict