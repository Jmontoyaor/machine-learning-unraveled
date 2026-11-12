import argparse

from mnist_app.training import train


def main():
    parser = argparse.ArgumentParser(description="Train one reference MNIST model.")
    parser.add_argument("--kind", choices=("dense", "cnn"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, help="Optional training subset for a quick smoke run.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.samples is not None and args.samples < 1):
        parser.error("epochs, batch size, and samples must be positive")
    train(args.kind, args.epochs, args.batch_size, args.seed, args.samples)


if __name__ == "__main__":
    main()
