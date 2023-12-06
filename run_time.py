import os
import time  
import torch
import torch.nn as nn
import numpy as np
import src.conv_onet.models.decoder as decoder
from torchsummary import summary
import src.config as config

HIDDEN_SIZE = 20
print(f"hidden size: {HIDDEN_SIZE}")

from pretrained_slim import load_pretrain, load_bound

cfg = config.load_config("./configs/Replica/room0.yaml", "./configs/nice_slam.yaml")

nice = decoder.NICE(coarse=True).to(cfg["mapping"]["device"])
slim_nice = decoder.NICE(hidden_size=HIDDEN_SIZE, coarse=True).to(cfg["mapping"]["device"])


load_pretrain(nice, cfg)
load_bound(nice, cfg)
load_bound(slim_nice, cfg)

stages = ["coarse", "middle", "fine", "color"]

time_record = {stage: [] for stage in stages}
slim_time_record = {stage: [] for stage in stages}
backward_time_record = {stage: [] for stage in stages}
slim_backward_time_record = {stage: [] for stage in stages}

for stage in stages:
    for i, pt in enumerate(os.listdir(f"./saved_inputs/{stage}")[:100]):
        data = torch.load(f"./saved_inputs/{stage}/{pt}")
        pi = data["pi"].to(cfg["mapping"]["device"])
        c = data["c"]

        # Forward pass for pretrained model
        time1 = time.time()
        pretrained_result = nice(pi, c_grid=c, stage=stage)
        time2 = time.time()
        time_record[stage].append(time2 - time1)

        # Backward pass for pretrained model
        pretrained_result.sum().backward()  # assuming pretrained_result is not a scalar
        time3 = time.time()
        backward_time_record[stage].append(time3 - time2)

        # Clear gradients
        if pretrained_result.grad is not None:
            pretrained_result.grad.data.zero_()

        # Forward pass for slim model
        time1 = time.time()
        slim_result = slim_nice(pi, c_grid=c, stage=stage)
        time2 = time.time()
        slim_time_record[stage].append(time2 - time1)

        # Backward pass for slim model
        slim_result.sum().backward()  # assuming slim_result is not a scalar
        time3 = time.time()
        slim_backward_time_record[stage].append(time3 - time2)

        # Clear gradients
        if slim_result.grad is not None:
            slim_result.grad.data.zero_()


for stage in stages:
    print(f"stage: {stage}")
    
    print(f"pretrained forward mean: {np.mean(time_record[stage])}")
    print(f"pretrained backward mean: {np.mean(backward_time_record[stage])}")
    
    print(f"slim forward mean: {np.mean(slim_time_record[stage])}")
    print(f"slim backward mean: {np.mean(slim_backward_time_record[stage])}")
    
    print(f"forward ratio: {np.mean(slim_time_record[stage]) / np.mean(time_record[stage])}")
    print(f"backward ratio: {np.mean(slim_backward_time_record[stage]) / np.mean(backward_time_record[stage])}")



