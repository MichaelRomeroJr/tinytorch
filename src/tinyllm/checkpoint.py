
import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    vocab_size
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "config": config,
    }, path)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint
