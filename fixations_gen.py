import torch
import torch.nn as nn
import sys

import gaussian_mask as G
import neural_population as CW
import foveated_image as FI
import vgg_backbone

class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, activation=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(c_in, 
                            c_out, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding, 
                            dilation=dilation, 
                            bias=bias, 
                            padding_mode='reflect')
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.relu(self.bn(self.conv(out))) if self.activation else self.conv(out)
        return out


class AgentNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.softmax = nn.Softmax(1)
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.cnn = vgg_backbone.CNNModel(args.n_classes)
        self.conv5 = Conv2d(512+256, 64, kernel_size=3, padding=1, activation=True, bias=True) 
        self.conv6 = Conv2d(64, 1, kernel_size=1, padding=0, activation=False, bias=True) 
        self.GF = FI.Get_foveated_images([1,  3,  5], [40,  70,  90], kss=7, device='cuda')


    def forward(self, img, ior, step, fixs_xy):
        ''' 2022/1/30
        input:
            img: (b, 3, 224, 224), tensor. image. 
            ior: (b, 1, h, w), inhibition of return in Cartesian form.
            step: int, current step index
            fixs_xy: (b, 2), float, ranged -1~1, location of fixation from previous step
                step1: Dorsal fix1 ---> Ventral fix2
                step2: Dorsal fix2 ---> Ventral fix3
                step3: Dorsal fix3 ---> Ventral fix4
                The fist fix1 for dorsal is random or center. 

        return:
            fix_next_xs, fix_next_ys: (b,), in range -1~1
        '''
        args = self.args

        if args.useWindow:
            ######################################
            # Consider the image img is already padded and is in square shape.
            # For images that is already in square shape, window is not needed. 
            # Window is needed when the input images are not in square shape. 
            # For example, movie frames are not in the square shape, and they are padded to be square from the data loader.  
            # The window is needed to describe image regions. 
            #####################################
            image_region = (args.image_size_h, args.image_size_w) #(126, 224)
            n_pad = int((image_region[1]-image_region[0])/2)
            window = torch.zeros((img.size(0), 1, img.size(-2), img.size(-1)), device='cuda')
            window[:, : ,n_pad-1:img.size(-2)-n_pad, :] = window[:, : ,n_pad-1:img.size(-2)-n_pad, :] + 1.0
            grid_forward = CW.make_xy2ret_grid_r(fixs_xy, window.size(-1), args.actmap_s, args.density_ratio_D)
            window_warp = torch.nn.functional.grid_sample(window, grid_forward, align_corners=True, mode='bilinear')




        
        ######################################
        # Feature extraction and Activation map
        #####################################
        # forward warping of input images
        grid_forward = CW.make_xy2ret_grid_r(fixs_xy, img.size(-1), args.grid_size_D, args.density_ratio_D)
        img_fov = self.GF(img, fixs_xy)
        img_warp = torch.nn.functional.grid_sample(img_fov, grid_forward, align_corners=True, mode='bilinear')
        # CNN feature extraction
        out4, out3, dict_D = self.cnn(img_warp)
        
        # Concatenate features from 4th and 3rd layers
        args.actmap_s = out4.size(-1)
        out3s = torch.nn.functional.interpolate(out3, (args.actmap_s, args.actmap_s), mode='bilinear')
        out5 = torch.cat((out3s, out4), 1)
        # activation map in warp space
        out_conv5 = self.conv5(out5)
        actmap_warp = self.conv6(out_conv5)
        actmap_warp = self.tanh(actmap_warp) 
        
        if args.useWindow:
            # actmap_warp ranges from -1 to 1
            # window_warp ranges from  0 to 1
            actmap_warp = (actmap_warp + 1) / 2.0
            # actmap_warp ranges from  0 to 1
            actmap_warp = actmap_warp * window_warp
        dict_D['feature_D_actmap_warp'] = actmap_warp
        dict_D['feature_D_out_conv5'] = out_conv5

        ######################################
        # IOR (Inhibition of Return)
        #####################################
        amap_s = actmap_warp.size()
        grid_forward = CW.make_xy2ret_grid_r(fixs_xy, ior.size(-1), args.actmap_s, args.density_ratio_D)
        ior_warp = torch.nn.functional.grid_sample(ior, grid_forward, align_corners=True, mode='bilinear')
        dict_D['feature_D_ior_warp'] = ior_warp

        # Activation map inhibited by IOR
        if args.useIOR:
            # Use IOR
            actmap_ior_warp = actmap_warp * ior_warp
        else:
            # Skip IOR
            actmap_ior_warp = actmap_warp
        actmap_ior_sm_warp = self.softmax(actmap_ior_warp.view(amap_s[0], -1)).view_as(ior_warp) # (batch 1, h, w) 
        actmap_sm_warp = self.softmax(actmap_warp.view(amap_s[0], -1)).view_as(ior_warp) # (batch 1, h, w) 
        dict_D['feature_D_actmap_ior_warp'] = actmap_ior_warp
        dict_D['feature_D_actmap_ior_sm_warp'] = actmap_ior_sm_warp

        ior_sigmas = torch.ones((amap_s[0],1), device='cuda') * 25

        ## Sample a point from warped HM
        dist = torch.distributions.categorical.Categorical(actmap_ior_sm_warp.view(amap_s[0], -1))

        if args.useStocFixs:
            # Sample a fixation point from the activation map
            idx = dist.sample() # (b, 1), index based one dim coordinate, 0~w*h.
        else:
            # Just take the maximum location from the activation map, without sampling. 
            idx = torch.max(actmap_ior_sm_warp.view(amap_s[0], -1), 1)[1]
        

        #############################
        # Data for Reinforcement learning
        #############################
        dist_orig = torch.distributions.categorical.Categorical(actmap_sm_warp.view(amap_s[0], -1))
        attn_log_pi = dist_orig.log_prob(idx)

        
        ######################################
        # Calculate fixation point in the retinal space (warped space) x'y'
        #####################################
        res_w = amap_s[-1]
        res_h = amap_s[-2]
        fix_next_ys, fix_next_xs = idx//res_w, idx%res_w # (b, 2), pixel level index based on two dim coords,
        fix_next_ys = (fix_next_ys.type(torch.float32) / float(res_h) - 0.5) * 2.0 * 0.95
        fix_next_xs = (fix_next_xs.type(torch.float32) / float(res_w) - 0.5) * 2.0 * 0.95
        fix_next_xy = torch.cat((fix_next_xs.unsqueeze(1), fix_next_ys.unsqueeze(1)), 1)
        ## This is x'y' space


        ######################################
        # Coordinates transform from x'y' to xy 
        # xy space is the regular Cartesian space without warping. 
        #####################################
        xy = CW.convert_coords_ret2xy_r(fixs_xy, fix_next_xy, img.size(-1), img.size(-1), args.density_ratio_D)
        fix_next_xs = xy[:,0]
        fix_next_ys = xy[:,1]



        ######################################
        # inverse warping ONLY FOR VISUALIZATION
        #####################################
        grid_inverse = CW.make_ret2xy_grid_r(fixs_xy, args.actmap_s, args.actmap_s, args.density_ratio_D)
        actmap = torch.nn.functional.grid_sample(actmap_warp, grid_inverse, align_corners=True)
        actmap_ior = torch.nn.functional.grid_sample(actmap_ior_warp, grid_inverse, align_corners=True)
        actmap_ior_sm = torch.nn.functional.grid_sample(actmap_ior_sm_warp, grid_inverse, align_corners=True)
        dict_D['feature_D_actmap'] = actmap
        dict_D['feature_D_actmap_ior'] = actmap_ior
        dict_D['feature_D_actmap_ior_sm'] = actmap_ior_sm
        if args.useWindow:
            dict_D['feature_D_actmap_warp'] = actmap_warp
            dict_D['feature_D_actmap_unwarp'] = actmap
            dict_D['window_warp'] = window_warp
            dict_D['window'] = window


        return fix_next_xs, fix_next_ys, actmap, actmap_ior, actmap_ior_sm, attn_log_pi, ior_sigmas, actmap_warp, img_warp, actmap_ior_warp, actmap_ior_sm_warp, dict_D
