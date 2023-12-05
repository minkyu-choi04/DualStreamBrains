import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
import os
import argparse
import sys

import vgg_backbone as retina
import misc
import gaussian_mask as G
import neural_population as CW
import foveated_image as FI

import fixations_gen

class CRNN_Model(nn.Module):
    def __init__(self, args, isFixedFixs, isRandomFixs=False):
        super().__init__()

        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, args.n_classes)
        self.gru = nn.GRU(input_size=512, hidden_size=512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = misc.Flatten()
        self.tanh = nn.Tanh()
        
        self.agent_net = fixations_gen.AgentNet(args)
        self.retina_net = retina.CNNModel(args.n_classes)
        self.GF = FI.Get_foveated_images([1,  2,  3], [40,  70,  90], kss=7, device='cuda')

        self.init_hidden = nn.Parameter(torch.randn((1, 1, 512)), requires_grad=True)
        self.isFixedFixs = isFixedFixs
        # added 2022.11.08
        self.isRandomFixs = isRandomFixs
        if self.isFixedFixs == True and self.isRandomFixs == True:
            raise RuntimeError(f'self.isFixedFixs == True and self.isRandomFixs == True')

    def forward(self, img, isTrain=True, n_steps=1, n_frames=None, dict_prev=None):
        args = self.args
        if n_steps == None:
            n_steps = args.n_steps
        
        batch_s = img.size(0)
        return_dict = {}
        return_dict['fixs_xy'] = []
        return_dict['fixs_xy_D'] = []
        return_dict['ior'] = []
        return_dict['img_fov'] = []
        return_dict['img_warp'] = []
        return_dict['img_warp_D'] = []
        return_dict['pred'] = []
        return_dict['hm'] = []
        return_dict['log_pi'] = []
        return_dict['actmap'] = []
        return_dict['actmap_ior'] = []
        return_dict['actmap_ior_sm'] = []
        return_dict['actmap_warp'] = []
        return_dict['actmap_ior_warp'] = []
        return_dict['actmap_ior_sm_warp'] = []


        for step in range(n_steps):
            '''
            Fixations
            Step 0: fix0 = Dorsal(random or center fixation), Ventral(fix0)
            Step 1: fix1 = Dorsal(fix0), Ventral(fix1)
            Step 2: fix2 = Dorsal(fix1), Ventral(fix2)
            Step 3: fix3 = Dorsal(fix2), Ventral(fix3)
            Step 4: fix4 = Dorsal(fix3), Ventral(fix4)
            Look at the relationship between fixations in D/V. 
            Dorsal fixations (produced from the previous step's fixation) is different from Ventral fixation. 
            '''

            if n_frames == 0:
                if isTrain:
                    fixs_x = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
                    fixs_y = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9
                else:
                    fixs_x = torch.zeros((batch_s,), dtype=torch.float32, device='cuda')
                    fixs_y = torch.zeros((batch_s,), dtype=torch.float32, device='cuda')
                fixs_xy = torch.cat((fixs_x.unsqueeze(1), fixs_y.unsqueeze(1)), 1)
                # This is only used for the dorsal's initial fixation. 
                
                ior = torch.ones((batch_s, 1, 224, 224), dtype=torch.float32, device='cuda')
                rnn_state = self.tanh(self.init_hidden.repeat(1, batch_s, 1))
            else:
                rnn_state = dict_prev['feature_V_rnn_output']
                fixs_xy = dict_prev['fixs_xy']
                ior = dict_prev['ior']

            inv_decay_ior = (1-ior) * 0.631
            ior = 1 - inv_decay_ior


            if self.isFixedFixs:
                fixs_xy = torch.zeros((batch_s, 2), device='cuda')
                

            # Dorsal Fixations (from previous step)
            return_dict['fixs_xy_D'].append(fixs_xy)
            
            # Ventral Fixations (from current step's dorsal)
            fixs_x, fixs_y, actmap, actmap_ior, actmap_ior_sm, attn_log_pi, ior_sigmas, actmap_warp, img_warp_D, actmap_ior_warp, actmap_ior_sm_warp, dict_D = self.agent_net(img, ior, step, fixs_xy)
            fixs_x, fixs_y = fixs_x.detach(), fixs_y.detach()
                
            fixs_xy = torch.cat((fixs_x.unsqueeze(1), fixs_y.unsqueeze(1)), 1)

            if self.isFixedFixs:
                fixs_xy = torch.zeros((batch_s, 2), device='cuda')

            # added 2022.11.08
            if self.isRandomFixs:
                size_img_h = args.image_size_h
                size_img_w = args.image_size_w
                assert size_img_h <= size_img_w, f'image size must be h<=w'
                ratio_img_h = size_img_h / size_img_w
                assert ratio_img_h != 0.0 , f'ratio_img_h must not be 0'
                fixs_x = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * 0.9 
                fixs_y = (torch.rand((batch_s,), dtype=torch.float32, device='cuda') - 0.5)*2.0 * ratio_img_h * 0.9 
                fixs_xy = torch.cat((fixs_x.unsqueeze(1), fixs_y.unsqueeze(1)), 1)
                
            return_dict['fixs_xy'].append(fixs_xy)
            # added 2022.11.08
            dict_D['fixs_xy'] = fixs_xy

            ior =  G.get_gaussian_mask(fixs_xy, mask_prev=ior, heatmap_s=(224, 224), sigma=ior_sigmas) 
            
            ## for visualization purpose
            ior_ones = torch.ones((batch_s, 1, 224, 224), dtype=torch.float32, device='cuda')
            ior_ones_temp =  1 - G.get_gaussian_mask(fixs_xy, mask_prev=ior_ones, heatmap_s=[224, 224])
            ior_curr_only =  torch.nn.functional.interpolate(ior_ones_temp, 
                    (18, 18), mode='bilinear')
                
            ## Warp Cart images
            grid_forward = CW.make_xy2ret_grid_r(fixs_xy, img.size(-1), args.grid_size_V, args.density_ratio_V)
            img_fov = self.GF(img, fixs_xy)
            img_warp = torch.nn.functional.grid_sample(img_fov, grid_forward, align_corners=True, mode='bilinear')

            feature_last, _, dict_V = self.retina_net(img_warp, stream='V')
            rnn_input = self.flat(self.gap(feature_last))
            self.gru.flatten_parameters()
            rnn_out, _ = self.gru(rnn_input.unsqueeze(0), rnn_state)
            pred = self.fc(rnn_out.squeeze(0))

            rnn_state = rnn_out

            hm = torch.squeeze(actmap) 

            # loss
            return_dict['log_pi'].append(attn_log_pi)

            # visualization data
            return_dict['ior'].append(ior)
            return_dict['img_fov'].append(img_fov)
            return_dict['img_warp'].append(img_warp)
            return_dict['img_warp_D'].append(img_warp_D)
            return_dict['pred'].append(pred)
            return_dict['hm'].append(hm)
            return_dict['actmap'].append(misc.normalize_min_max(actmap))
            return_dict['actmap_ior'].append(misc.normalize_min_max(actmap_ior)) 
            return_dict['actmap_ior_sm'].append(misc.normalize_min_max(actmap_ior_sm))
            return_dict['actmap_warp'].append(actmap_warp)
            return_dict['actmap_ior_warp'].append(misc.normalize_min_max(actmap_ior_warp))
            return_dict['actmap_ior_sm_warp'].append(misc.normalize_min_max(actmap_ior_sm_warp))

            dict_V['feature_V_rnn_input'] = rnn_input
            dict_V['feature_V_rnn_output'] = rnn_out
            dict_V['feature_V_fc'] = pred

        return return_dict, dict_V, dict_D


