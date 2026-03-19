from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def ddr_mvc_loss(
    reconstructions: List[torch.Tensor],
    inputs: List[torch.Tensor],
    z_views: List[torch.Tensor],
    z_fused: torch.Tensor,
    alpha_rec: float,
    alpha_con: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DDR-MVC objective: reconstruction + view-to-fusion consistency."""
    rec = sum(F.mse_loss(recon, x) for recon, x in zip(reconstructions, inputs))
    con = sum(F.mse_loss(z_view, z_fused) for z_view in z_views) / len(z_views)
    total = alpha_rec * rec + alpha_con * con
    return total, {"loss": float(total.item()), "rec": float(rec.item()), "con": float(con.item())}
