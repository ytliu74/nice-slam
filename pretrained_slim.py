import torch
import numpy as np
import src.conv_onet.models.decoder as decoder
from torchsummary import summary
import src.config as config


def load_pretrain(nice: decoder.NICE, cfg: dict):
    ckpt = torch.load(cfg['pretrained_decoders']['coarse'], map_location=cfg['mapping']['device'])
    coarse_dict = {}
    for key, val in ckpt['model'].items():
        if ('decoder' in key) and ('encoder' not in key):
            key = key[8:]
            coarse_dict[key] = val
    nice.coarse_decoder.load_state_dict(coarse_dict)

    ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'], map_location=cfg['mapping']['device'])
    middle_dict = {}
    fine_dict = {}
    for key, val in ckpt['model'].items():
        if ('decoder' in key) and ('encoder' not in key):
            if 'coarse' in key:
                key = key[8+7:]
                middle_dict[key] = val
            elif 'fine' in key:
                key = key[8+5:]
                fine_dict[key] = val
    nice.middle_decoder.load_state_dict(middle_dict)
    nice.fine_decoder.load_state_dict(fine_dict)

    scale = cfg['scale']

    bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*scale)
    bound_divisible = cfg['grid_len']['bound_divisible']
    bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
        bound_divisible).int()+1)*bound_divisible+bound[:, 0]

    nice.bound = bound
    nice.middle_decoder.bound = bound
    nice.fine_decoder.bound = bound
    nice.color_decoder.bound = bound
    nice.coarse_decoder.bound = bound * cfg['model']['coarse_bound_enlarge']


nn = decoder.NICE(coarse=True).to("cuda:0")
cfg = config.load_config("./configs/Replica/room0.yaml", "./configs/nice_slam.yaml")

load_pretrain(nn, cfg)

import os

stages = ["coarse", "middle", "fine", "color"]

for stage in stages:
    for pt in os.listdir(f"./saved_inputs/{stage}")[:1]:
        data = torch.load(f"./saved_inputs/{stage}/{pt}")
        pi = data['pi'].to(cfg['mapping']['device'])
        c = data['c']
        print(nn(pi, c_grid=c, stage=stage))