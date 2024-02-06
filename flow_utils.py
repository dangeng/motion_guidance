'''
Utilities for flow prediction and manipulation
'''
import pathlib
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFT(nn.Module):
    def __init__(self, model='things', num_iters=5, dropout=0):
        super(RAFT, self).__init__()
        
        from flow_models.raft import raft

        if model == 'things':
            model = 'raft-things.pth'
        elif model == 'kitti':
            model = 'raft-kitti.pth'
        elif model == 'sintel':
            model = 'raft-sintel.pth'

        # Get location of checkpoints
        raft_dir = pathlib.Path(__file__).parent.absolute()/'flow_models'/'raft'

        # Emulate arguments
        args = argparse.Namespace()
        args.model = raft_dir / model
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False
        args.dropout = dropout

        flowNet = nn.DataParallel(raft.RAFT(args))
        flowNet.load_state_dict(torch.load(args.model, map_location='cpu'))
        self.flowNet = flowNet.module.cpu()

        self.num_iters = num_iters

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''

        # Normalize to [0, 255]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        flow_low, flow_up = self.flowNet(im1, im2, iters=self.num_iters, test_mode=True)

        return flow_up

def normalize_flow(flow):
    '''
    Normalize pixel-offset (relative) flow to absolute [-1, 1] flow
    input :
        flow : tensor (b, 2, h, w)
    output :
        flow : tensor (b, h, w, 2) (for `F.grid_sample`)
    '''
    _, _, h, w = flow.shape
    device = flow.device

    # Get base pixel coordinates (just "gaussian integers")
    base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
    base = torch.stack(base).float().to(device)

    # Convert to absolute coordinates
    flow = flow + base

    # Convert to [-1, 1] for grid_sample
    size = torch.tensor([w, h]).float().to(device)
    flow = -1 + 2.*flow/(-1 + size)[:,None,None]
    flow = flow.permute(0,2,3,1)
    
    return flow
    
def warp(im, flow, padding_mode='reflection'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''

    warped = F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped