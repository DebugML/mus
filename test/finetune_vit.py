from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as tvtfs
from torch.utils.data import Dataset, DataLoader
import datasets as hfds
from transformers import ViTForImageClassification, get_scheduler
from tqdm import tqdm


def load_model(model_name_or_path: str = "google/vit-base-patch16-224-in21k"):
    return ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=1000)


class ImagenetDataset(Dataset):
    """ Imagenet dataset """
    def __init__(self, split: str, rand_masks: bool = False):
        self.ds = hfds.load_dataset("ILSVRC/imagenet-1k", split=split)
        self.preprocessor = tvtfs.Compose([
            tvtfs.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            tvtfs.Resize(256, interpolation=tvtfs.InterpolationMode.BICUBIC),
            tvtfs.RandomCrop(224),
            tvtfs.ToTensor(),
            tvtfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.rand_masks = rand_masks

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = self.preprocessor(item["image"])

        # Each 16x16 patch (14x14 grid) has a random chance of turning on, randomly.
        if self.rand_masks:
            small_mask = torch.rand(1,1,14,14) < torch.rand(())
            large_mask = F.interpolate(small_mask.float(), scale_factor=(16,16)).float()
            image = image.view(3,224,224) * large_mask.view(1,224,224)
        return {
            "image": image,
            "label": item["label"]
        }


def train_one_epoch(model, dataloader, optimizer, lr_scheduler):
    model.train()
    device = next(model.parameters()).device
    all_losses, all_accs = [], []
    pbar = tqdm(dataloader)
    for batch in pbar:
        image, label = batch["image"].to(device), batch["label"].to(device)
        logits = model(image).logits
        loss = F.cross_entropy(logits, label)
        loss.backward(); optimizer.step(); optimizer.zero_grad(); lr_scheduler.step()

        all_losses.append(loss.detach().item())
        all_accs.append((logits.argmax(dim=-1) == label).float().mean().detach().item())
        pbar.set_description(
            f"lr {lr_scheduler.get_last_lr()[0]:.3e}, "
            + f"loss {torch.tensor(all_losses[-16:]).mean():.3e}, "
            + f"acc {torch.tensor(all_accs[-16:]).float().mean():.3e}"
        )

    return {
        "model_state_dict": model.state_dict(),
        "train_losses": all_losses,
        "train_accs": all_accs,
    }


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    all_losses, all_accs = [], []
    pbar = tqdm(dataloader)
    for batch in pbar:
        image, label = batch["image"].to(device), batch["label"].to(device)
        logits = model(image).logits
        loss = F.cross_entropy(logits, label)

        all_losses.append(loss.detach().item())
        all_accs.append((logits.argmax(dim=-1) == label).float().mean().detach().item())
        pbar.set_description(
            f"loss {torch.tensor(all_losses[-16:]).mean():.3e}, "
            + f"acc {torch.tensor(all_accs[-16:]).float().mean():.3e}"
        )

    return {
        "valid_losses": all_losses,
        "valid_accs": all_accs
    }


def run_finetuning_epochs(
    num_epochs: int = 5,
    bsz: int = 128,
    lr: float = 1e-4,
    device: str = "cuda",
    saveto_dir: str = "_saved_models",
):
    # Make directory if doesn't exist
    Path(saveto_dir).mkdir(parents=True, exist_ok=True)

    # Initialize dataloaders
    train_dataloader = DataLoader(ImagenetDataset("train", rand_masks=True), batch_size=bsz)
    valid_dataloader = DataLoader(ImagenetDataset("validation", rand_masks=True), batch_size=bsz)

    # Set up the model, optimizer, and learning rate scheduler
    model = load_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_train_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        name = "linear",
        optimizer = optimizer,
        num_warmup_steps = int(0.1 * num_train_steps),
        num_training_steps = num_train_steps
    )

    # Fine-tuning loop
    for epoch in range(1, num_epochs+1):
        saveto = f"{saveto_dir}/google_vit_patch16_img224_bsz{bsz}_lr{lr:.4f}_epoch{epoch}.pt" 
        print("Training", saveto)
        train_ret = train_one_epoch(model, train_dataloader, optimizer, lr_scheduler)
        valid_ret = valid_one_epoch(model, valid_dataloader)
        train_ret["valid_losses"] = valid_ret["valid_losses"]
        train_ret["valid_accs"] = valid_ret["valid_accs"]
        torch.save(train_ret, saveto)

    return train_ret


if __name__ == "__main__":
    run_finetuning_epochs()

