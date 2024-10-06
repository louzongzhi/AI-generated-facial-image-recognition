import torch


LOSSES = {
    "binary_ce": torch.nn.BCEWithLogitsLoss(),
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "mse": torch.nn.MSELoss()
}


def get_loss(name="cross_entropy", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using loss: '{LOSSES[name]}'")
    return LOSSES[name].to(device)
