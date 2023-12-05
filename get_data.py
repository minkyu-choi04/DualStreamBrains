import sys
import os
import torch
import time
import numpy as np

import misc

def save_debug_images(dir_out, inputs_time, return_dict, dict_D, n_frames, useWindow, idx_seg=0):
    misc.make_dir(os.path.join(dir_out, 'plots'))
    n_plots = 1 
    output_epoch = 0
    batch_i = 0

    if useWindow:
        fixs_until = torch.stack(return_dict['fixs_xy'][:0+1]).permute(1, 0, 2) # (b, step, 2)
        img_fixs = misc.mark_fixations_history(inputs_time[:n_plots]*dict_D['window'][:n_plots], fixs_until[:n_plots])
        
        hm_on_img = misc.add_heatmap_on_image_tensor(return_dict['hm'][0].unsqueeze(0), inputs_time[:n_plots]*dict_D['window'][:n_plots], resize_s=(224,224), device='cuda')

        images_total = misc.concatenate_images_horizontally(inputs_time[:n_plots]*dict_D['window'][:n_plots], img_fixs, hm_on_img, 128*3, 20)
        misc.plot_one_sample_from_images(images_total,  
                os.path.join(dir_out, 'plots'), 'output'+str(n_frames)+'.png', isRange01=True)

def get_data_HW(model: torch.nn.Module, 
                input_img_size: [int]=[174, 336], 
                useWindow: bool=False,
                pad_mod: str='replicate',
                n_zeros: int=5
                ) -> None:
    """Collect features from the given model
    Args: 
        input_img_size: size of input images to be loaded. The original frame size from dvd is (540 x 1046) for Raiders. 
                        Setting 1046 as 336, corresponding size of 540 becomes 174. 
                        If image is not in square shape (like movie Raiders or twilight), the movie frames will first be
                        padded to be square size of input_img_size[1] x input_img_size[1]. 
                        input_img_size[1] > input_img_size[0] is assumed and must always be met. 
        n_zeros: int, the number of leading zeros in filename in frames. Default: 5. 

    Returns:
        None
    """
    model.eval() 

    get_image = misc.Load_PIL_to_tensor(img_s_load=input_img_size)
    input_img_size_max = max(input_img_size)
    n_pad = int((input_img_size[1]-input_img_size[0])/2)
    numlist = [0, 1, 2] 
    
    n_frames = 0
    dict_prev = None
    for im in numlist:
        image = get_image.load_image_to_tensor(os.path.join('./input_images/frame'+str(im)+'.png'))
        image = image.cuda()
        transformed_image = torch.nn.functional.pad(image, (0,0,n_pad,n_pad), mode=pad_mod)
    
        with torch.no_grad():
            return_dict, out_V, out_D = model(transformed_image, isTrain=False, n_frames=n_frames, dict_prev=dict_prev)
            dict_prev = {}
            dict_prev['ior'] = return_dict['ior'][0]
            dict_prev['fixs_xy'] = return_dict['fixs_xy'][0]
            dict_prev['feature_V_rnn_output'] = out_V['feature_V_rnn_output']
            n_frames += 1

            save_debug_images('./', transformed_image, return_dict, out_D, n_frames, useWindow)
