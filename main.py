"""
Entry point for the MAR20 ship segmentation project.

Usage:
    python main.py train       # train the model
    python main.py evaluate    # run inference on test set
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|evaluate]")
        print("  train     - Train the UNet model")
        print("  evaluate  - Run inference on test set")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        import training  # noqa: F401 — runs training.__main__

    elif command == "evaluate":
        import result_script
        result_script.main()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train|evaluate]")
        sys.exit(1)


if __name__ == "__main__":
    main()
