#!/usr/bin/env python3
import os
import shutil
from pathlib import Path


def cleanup_training_files():
    current_dir = Path.cwd()
    models_dir = current_dir / "models"
    loss_dir = current_dir / "training_logs" / "loss"
    rewards_dir = current_dir / "training_logs" / "rewards"

    models_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    rewards_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for model_file in current_dir.glob("model*.pth"):
        if model_file.is_file():
            dest = models_dir / model_file.name
            print(f"Moving {model_file.name} -> models/{model_file.name}")
            shutil.move(str(model_file), str(dest))
            moved_count += 1

    for loss_file in current_dir.glob("training_loss_*.csv"):
        if loss_file.is_file():
            dest = loss_dir / loss_file.name
            print(f"Moving {loss_file.name} -> training_logs/loss/{loss_file.name}")
            shutil.move(str(loss_file), str(dest))
            moved_count += 1

    for rewards_file in current_dir.glob("training_rewards_*.csv"):
        if rewards_file.is_file():
            dest = rewards_dir / rewards_file.name
            print(f"Moving {rewards_file.name} -> training_logs/rewards/{rewards_file.name}")
            shutil.move(str(rewards_file), str(dest))
            moved_count += 1

    if moved_count == 0:
        print("No files to move. Current directory is clean!")
    else:
        print(f"\nSuccessfully moved {moved_count} file(s).")


if __name__ == "__main__":
    cleanup_training_files()
