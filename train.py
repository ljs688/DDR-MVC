import argparse

from ddr_mvc.config import DDRMVCConfig
from ddr_mvc.trainer import train_ddr_mvc


def parse_args():
    parser = argparse.ArgumentParser(description="Train the minimal DDR-MVC model.")
    parser.add_argument("--dataset-name", default="MSRCV1", help="Dataset name without .mat suffix")
    parser.add_argument("--data-dir", default="data", help="Directory containing dataset mat files")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--feature-dim", type=int, default=128, help="Latent feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--alpha-rec", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--alpha-con", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DDRMVCConfig(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        alpha_rec=args.alpha_rec,
        alpha_con=args.alpha_con,
        seed=args.seed,
    )

    print(f"[DDR-MVC] Project: {cfg.project_name}")
    print(f"[DDR-MVC] Dataset: {cfg.dataset_name}")
    print(f"[DDR-MVC] Device: {cfg.device}")

    metrics, checkpoint_path = train_ddr_mvc(cfg)
    print(
        "[DDR-MVC] Final metrics | "
        f"ACC={metrics['acc']:.4f} "
        f"NMI={metrics['nmi']:.4f} "
        f"ARI={metrics['ari']:.4f}"
    )
    print(f"[DDR-MVC] Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
