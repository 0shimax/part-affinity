import os
from pathlib import Path
import scipy.stats

import time

import cv2
import numpy as np
import torch
from data_process.process_utils import resize_hm, denormalize
from visualization.visualize import visualize_output_single
from .post import decode_pose, append_result


# Typical evaluation is done on multi-scale and average across all evals is taken as output
# These reduce the quantization error in the model
def test_net(data_loader, model, opts):
    model.eval()
    dataset = data_loader.dataset
    outputs = []
    dataset_len = 100 #len(dataset)
    runtimes = []
    with torch.no_grad():
        for i, imgs in enumerate(dataset):
            print(i)
            start = time.time()
            n_imgs = len(imgs)
            heights = list(map(lambda x: x.shape[1], imgs))
            widths = list(map(lambda x: x.shape[2], imgs))
            img_basic = imgs[0]

            heatmap_avg_lst = []
            paf_avg_lst = []
            print("first loop", time.time() - start)
            for j in range(0, n_imgs):
                imgs_torch = torch.from_numpy(imgs[j:j+1]).float().to(opts.device)
                heatmaps, pafs = model(imgs_torch)
                heatmap = heatmaps[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                paf = pafs[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                heatmap = resize_hm(heatmap, (widths[0], heights[0]))
                paf = resize_hm(paf, (widths[0], heights[0]))
                heatmap_avg_lst += [heatmap]
                paf_avg_lst += [paf]
            heatmap_avg = sum(heatmap_avg_lst)/n_imgs
            paf_avg = sum(paf_avg_lst)/n_imgs

            print("second loop", time.time() - start)
            #visualize_output_single(img_basic, heatmap_t, paf_t, ignore_mask_t, heatmap_avg, paf_avg)
            img_basic = denormalize(img_basic)

            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            canvas, to_plot, candidate, subset = decode_pose(img_basic, param, heatmap_avg, paf_avg)

            final = time.time()-start
            runtimes += [final]
            print("both loops took ", final)
            append_result(0, subset, candidate, outputs)
            vis_path = Path(opts.saveDir, 'viz')
            vis_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path) + '/{}.png'.format(i), to_plot)
    print ("runtime statistics for all images")
