import numpy as np
import cv2
import torch
import torch._dynamo.eval_frame
import os

def preprocess(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a NumPy array, but got {type(image)}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 110))
    image = image[13:97, :]

    return image.astype(np.float32) / 255.0


def stack_frames(stacked_frames, new_frame, is_new_episode):
    frame = preprocess(new_frame)

    if is_new_episode:
        stacked_frames = [frame] * 4
    else:
        stacked_frames.append(frame)
        stacked_frames.pop(0)

    return np.stack(stacked_frames, axis=0), stacked_frames


def load_model(filepath="model.pth"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at {filepath}")

    print(f"Loading model from {filepath}...")

    # Handle compiled models and new PyTorch security settings
    try:
        # First try with safe loading
        model = torch.load(filepath, weights_only=True)
        print("Successfully loaded with weights_only=True")
    except Exception as e1:
        print(f"Weights-only loading failed: {e1}")
        try:
            # Try with weights_only=False
            model = torch.load(filepath, weights_only=False)
            print("Successfully loaded with weights_only=False")
        except Exception as e2:
            print(f"Standard loading failed: {e2}")
            print("Attempting force load with compiled model support...")
            # Force load with compiled model globals
            torch.serialization.add_safe_globals([torch._dynamo.eval_frame.OptimizedModule])
            model = torch.load(filepath, weights_only=False)
            print("Successfully force loaded compiled model")

    model.eval()
    return model


def save_model(model, filepath="model.pth"):
    # Save the original model if it's compiled
    if hasattr(model, '_orig_mod'):
        print("Saving original model (uncompiled) for compatibility")
        torch.save(model._orig_mod, filepath)
    else:
        torch.save(model, filepath)
    print(f"Model saved to {filepath}")
