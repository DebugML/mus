from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import skimage
import sklearn

# from .common import relabel_segments_by_proximity


def my_quickshift(
    image: torch.FloatTensor,
    kernel_size: float = 8.,
    max_dist: float = 100.,
    sigma: float = 5.,
    **kwargs
):
    # image is (C,H,W)
    assert image.ndim == 3
    device = image.device
    image = image.cpu()

    # We have to make the images 3-channel for quickshift
    if image.size(0) == 1:
        image_np = image.repeat(3,1,1).numpy().transpose(1,2,0)
    elif image.size(0) == 3:
        image_np = image.numpy().transpose(1,2,0)
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")

    # quickshift returns a (H,W) of numpy integers
    segs = skimage.segmentation.quickshift(
        image_np,
        kernel_size = kernel_size,
        max_dist = max_dist,
        sigma = sigma,
        **kwargs
    )
    return torch.tensor(segs).to(device)


class QuickshiftGroups(nn.Module):
    # Use quickshift to perform image segmentation
    def __init__(
        self,
        max_groups: int = 16,
        kernel_size: float = 8.,
        max_dist: float = 20.,
        sigma: float = 1.,
        flat: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.max_groups = max_groups
        self.flat = flat

    def forward(self, x):
        # x: (N,C,H,W)
        all_segs = []
        for image in x:
            image = image.cpu()

            segs = my_quickshift(
                image,
                kernel_size = self.kernel_size,
                max_dist = self.max_dist,
                sigma=self.sigma
            )

            segs = relabel_segments_by_proximity(segs)

            if segs.unique().numel() > self.max_groups:
                div_by = (segs.unique().max() + 1) / self.max_groups
                segs = (segs // div_by).long()

            all_segs.append(segs)

        all_segs = torch.stack(all_segs).long().to(x.device)

        if self.flat:
            return all_segs
        else:
            return F.one_hot(all_segs, num_classes=self.max_groups).permute(0,3,1,2) # (N,M,H,W)


class NeuralQuickshiftGroups(nn.Module):
    #
    def __init__(
        self,
        max_groups: int = 16,
        kernel_size: float = 3.,
        max_dist: float = 50.,
        sigma = 10.,
        flat: bool = False,
        feature_extractor: str | Callable = "resnet18"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.max_groups = max_groups
        self.flat = flat
        if feature_extractor == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1] + [nn.Flatten(1)]))
            self.feature_extractor.eval()
        else:
            self.feature_extractor = feature_extractor

    @torch.no_grad()
    def go(self, image):
        device = image.device
        C, H, W = image.shape
        segs = my_quickshift(
            image,
            kernel_size = self.kernel_size,
            max_dist = self.max_dist,
            sigma = self.sigma
        )
        num_init_groups = 8 * self.max_groups

        if segs.unique().max() + 1 >= self.max_groups:
            div_by = (segs.unique().max() + 1) / num_init_groups
            segs = (segs // div_by).long()

        seg_masks = F.one_hot(segs, num_classes=num_init_groups).permute(2,0,1).to(device) # (P,H,W)
        masked_images = seg_masks.view(-1,1,H,W) * image.view(1,C,H,W)  # (P,C,H,W)

        # If single-channel, then repeat
        if C == 1:
            masked_images = masked_images.repeat(1,3,1,1)

        features = self.feature_extractor(masked_images)

        kmeans = sklearn.cluster.KMeans(n_clusters=self.max_groups).fit(features.detach().cpu().numpy())
        labels = torch.tensor(kmeans.labels_).to(device)

        regrouped_segs = torch.zeros(H, W).to(device)
        for i in range(self.max_groups):
            regrouped_segs += ((labels == i).view(num_init_groups,1,1) * seg_masks * i).sum(dim=0)

        segs = relabel_segments_by_proximity(segs)
        return regrouped_segs.long()

    @torch.no_grad()
    def forward(self, x):
        all_segs = torch.stack([self.go(image) for image in x])
        
        if self.flat:
            return all_segs
        else:
            return F.one_hot(all_segs, num_classes=self.max_groups).permute(0,3,1,2) # (N,M,H,W)


