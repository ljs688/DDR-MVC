from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DDRMVCConfig
from .data import DDRMVCDataset
from .losses import ddr_mvc_loss
from .metrics import clustering_metrics
from .model import DDRMVCModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate(model: DDRMVCModel, dataloader: DataLoader, device: torch.device, n_clusters: int) -> Dict[str, float]:
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch_xs, batch_labels in dataloader:
            batch_xs = [x.to(device) for x in batch_xs]
            _, _, z_fused, _ = model(batch_xs)
            all_features.append(z_fused.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    acc, nmi, ari = clustering_metrics(features, labels, n_clusters)
    return {"acc": acc, "nmi": nmi, "ari": ari}


def train_ddr_mvc(cfg: DDRMVCConfig) -> Tuple[Dict[str, float], Path]:
    """Train the minimal DDR-MVC model and return final metrics + checkpoint path."""
    set_seed(cfg.seed)

    dataset = DDRMVCDataset(cfg.dataset_name, cfg.data_dir)
    batch_size = min(cfg.batch_size, len(dataset))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = DDRMVCModel(
        view_dims=dataset.get_dims(),
        hidden_dim=cfg.hidden_dim,
        feature_dim=cfg.feature_dim,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_interval = max(1, cfg.epochs // 10)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_con = 0.0
        steps = 0

        for batch_xs, _ in train_loader:
            batch_xs = [x.to(cfg.device) for x in batch_xs]
            reconstructions, z_views, z_fused, _ = model(batch_xs)

            loss, stats = ddr_mvc_loss(
                reconstructions=reconstructions,
                inputs=batch_xs,
                z_views=z_views,
                z_fused=z_fused,
                alpha_rec=cfg.alpha_rec,
                alpha_con=cfg.alpha_con,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += stats["loss"]
            epoch_rec += stats["rec"]
            epoch_con += stats["con"]
            steps += 1

        if epoch % log_interval == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[DDR-MVC] Epoch {epoch:04d}/{cfg.epochs:04d} "
                f"loss={epoch_loss / steps:.4f} "
                f"rec={epoch_rec / steps:.4f} "
                f"con={epoch_con / steps:.4f}"
            )

    metrics = _evaluate(
        model=model,
        dataloader=eval_loader,
        device=cfg.device,
        n_clusters=dataset.get_num_clusters(),
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ddr_mvc_{cfg.dataset_name}.pt"
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict(), "metrics": metrics}, checkpoint_path)

    return metrics, checkpoint_path
