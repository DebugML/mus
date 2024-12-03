import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def sample_mus_noise(
    dim: int,
    lambd: float,
    quant: int,
    v_seed: torch.FloatTensor | None = None,
    device: str = "cpu",
    return_dict: bool = False,
):
    """
    Make the mus noise.

    Args:
        dim: The dimension
        lambd: The drop prob (and Lipschitz const). Should be a multiple of 1/q.
        quant: The quantization parameter (q)
        v_seed: The seed noise
    """
    q = quant
    lambd = int(lambd * q) / q

    if v_seed is None:
        v_seed = torch.randint(0, q, (1, dim), device=device) / q
    
    s_base = ((torch.arange(q, device=device) + 0.5) / q).view(q,1)
    t = (v_seed + s_base).remainder(1.0) # (q, dim)
    s = (t < lambd).long()
    if return_dict:
        return {"mask": s, "pre_mask": t, "v_seed": v_seed}
    else:
        return s


class GridSegmenter(nn.Module):
    def __init__(self, grid_size: tuple[int,int] = (8,8)):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, x):
        C, H, W = x.shape
        gH, gW = self.grid_size
        pH = (H // gH) + (H % gH != 0)
        pW = (W // gW) + (W % gW != 0)
        mask_small = torch.arange(gH * gW, device=x.device).view(1, 1, gH, gH)
        mask_big = F.interpolate(mask_small.float(), scale_factor=(pH,pW)).round().long()
        return mask_big[0,0,:H,:W] # (H,W)


class VisionMuS(nn.Module):
    def __init__(
        self,
        model: nn.Module, 
        lambd: float,
        quant: int = 64,
        segment_fn: Callable = GridSegmenter(),
    ):
        super().__init__()
        self.model = model
        self.segment_fn = segment_fn
        self.q = quant
        self.lambd = int(lambd * self.q) / self.q

    def forward(self, xs: torch.FloatTensor, return_all: bool = False):
        bsz, C, H, W = xs.shape
        q, lambd = self.q, self.lambd
        all_seg_maps, all_ys = (), ()
        for x in xs:
            seg_map = self.segment_fn(x) # int-valued (H,W) with values 0, ..., num_segs - 1
            all_seg_maps += (seg_map,)

            num_segs = seg_map.max() + 1
            mus_mask = sample_mus_noise(num_segs, lambd, q, device=x.device) # (q,num_segs)
            big_mask = torch.matmul(
                F.one_hot(seg_map, num_classes=num_segs).view(1,H,W,1,num_segs).float(),
                mus_mask.view(q,1,1,num_segs,1).float()
            ).view(q,H,W).long()

            x_masked = x.view(1,C,H,W) * big_mask.view(q,1,H,W) # (q,C,H,W)
            ys = self.model(x_masked)   # (q,num_classes)

            # Convert to one-hot and vote
            ys = F.one_hot(ys.argmax(dim=-1), num_classes=ys.size(-1))  # (q, num_classes)
            avg_y = ys.float().mean(dim=0) # (num_classes)
            all_ys += (avg_y,)

        all_ys = torch.stack(all_ys) # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * lambd)

        return {
            "votes": all_ys, # (bsz, num_classes) 
            "seg_maps": torch.stack(all_seg_maps), # (bsz, H, W)
            "cert_rs": cert_rs # (bsz,)
        }


