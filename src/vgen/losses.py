import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from flow_utils import warp, normalize_flow
from flow_viz import flow_to_image
from PIL import Image

class FlowLoss(nn.Module):
    def __init__(self, color_weight=100.0, flow_weight=1.0, 
                 oracle=False, 
                 target_flow=None,
                 occlusion_masking=False):
        super().__init__()

        # Make flow network
        from flow_utils import RAFT
        self.flow_net = RAFT()

        # Get params
        self.flow_weight = flow_weight
        self.color_weight = color_weight
        self.oracle = oracle
        self.target_flow = target_flow
        self.occlusion_masking = occlusion_masking

        # Make color mask (is 0 where we want to mask) 
        # (this is just occluded areas)
        # We use `grad` to find occluded areas, which is nicer than
        # just forward splatting
        with torch.enable_grad():
            tgt_flow = self.target_flow.clone()
            tgt_flow.requires_grad = True
            warped_flow = warp(tgt_flow, normalize_flow(tgt_flow))
            masked_warped_flow = warped_flow * (tgt_flow != warped_flow)
            grad = torch.autograd.grad(masked_warped_flow.sum(), tgt_flow)[0]
            self.mask_occ = 1 - (grad.abs().sum(1) != 0).float()
            self.mask_occ = self.mask_occ[:,None]
            # But we don't want to mask out any original pixels! 
            # (we use non-zero flow as proxy for "original pixel")
            self.mask_occ[self.target_flow.abs().sum(1, keepdim=True) != 0] = 1

    def masked_l1(self, x, y, mask):
        mask = mask.to(x.device)
        x = x * mask
        y = y * mask
        return l1_loss(x, y)

    def forward(self, pred, target):
        device = pred.device

        # Normalize (inputs will be (approximately) [-1, 1])
        target = target / 2. + 0.5
        pred = pred / 2. + 0.5

        # Get flow prediction (detach disoccluded pixels)
        flow = self.flow_net(target, pred)

        # Compute flow loss
        flow_tgt = self.target_flow.to(device)
        flow_loss = l1_loss(flow_tgt, flow)

        # Compute color loss
        if self.oracle:
            pred_warped = warp(pred, normalize_flow(flow_tgt))
        else:
            pred_warped = warp(pred, normalize_flow(flow))
        # Optionally mask occlusions
        if self.occlusion_masking:
            color_loss = self.masked_l1(target, pred_warped, self.mask_occ)
        else:
            color_loss = l1_loss(target, pred_warped)

        # Compute full loss
        loss = self.flow_weight * flow_loss + self.color_weight * color_loss

        # Make info
        flow_im = Image.fromarray(flow_to_image(flow[0].permute(1,2,0).cpu().detach().numpy()))
        info = {}
        info['flow_loss'] = flow_loss.item()
        info['color_loss'] = color_loss.item()
        info['flow_im'] = flow_im

        return loss, info


