import pickle

import yaml
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os
import joblib
from datetime import datetime

try:
    from .models import SklearnAudioClassifier
    from .dataset import AcousticScenesDataset
except ImportError:
    from models import SklearnAudioClassifier
    from dataset import AcousticScenesDataset


def extract_all_features(model, dataloader, desc="Extracting features"):
    """Extract features with progress bar."""
    X_all, y_all = [], []
    for batch in tqdm(dataloader, desc=desc):
        audio = batch['audio_data']
        labels = batch['class_label'].numpy()
        X_all.append(model.extract_features(audio))
        y_all.append(labels)
    return np.vstack(X_all), np.concatenate(y_all)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('conf matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"saved conf matrix to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print(f"Training: {config['network']['type']}")
    print("=" * 50)

    # Load datasets
    train_dataset = AcousticScenesDataset(
        dataset_name="train",
        sample_rate=config["data"]["sample_rate"],
        mono=config["data"]["mono"],
        base_data_path=config["data"]["base_data_path"],
    )
    val_dataset = AcousticScenesDataset(
        dataset_name="val",
        sample_rate=config["data"]["sample_rate"],
        mono=config["data"]["mono"],
        base_data_path=config["data"]["base_data_path"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["n_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["n_workers"],
    )

    ckpt_dir = config.get("logging", {}).get("ckpt_dir", "./data/ckpts/sklearn/")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = SklearnAudioClassifier(
        classifier_type=config["network"]["type"],
        sample_rate=config["data"]["sample_rate"],
        n_mels=config["network"]["n_mels"],
        **config["network"].get("params", {})
    )

    print("\n[1/3] extracting training features..")
    X_train, y_train = extract_all_features(model, train_loader, "Train")

    print("\n[2/3] fitting..")
    model.pipeline.fit(X_train, y_train)

    print("\n[3/3] eval..")
    X_val, y_val = extract_all_features(model, val_loader, "Val")

    train_acc = (model.pipeline.predict(X_train) == y_train).mean()
    val_pred = model.pipeline.predict(X_val)
    val_acc = (val_pred == y_val).mean()

    # res
    print("\n" + "=" * 50)
    print("results:")
    print("=" * 50)
    print(f"train acc: {train_acc:.3f}")
    print(f"val acc:   {val_acc:.3f}")

    # classification report
    class_names = sorted(train_dataset.meta_df['scene_name'].unique())
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=class_names))

    plot_confusion_matrix(y_val, val_pred, class_names,
                          save_path=config.get("logging", {}).get("cm_path"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['network']['type']}_{timestamp}-val_acc={val_acc:.2f}"
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.ckpt")

    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'pipeline': model.pipeline,
            'mel_transform_config': {
                'sample_rate': config["data"]["sample_rate"],
                'n_mels': config["network"]["n_mels"],
            },
            'classifier_type': config['network']['type'],
            'val_accuracy': val_acc,
            'config': config,
        }, f)

if __name__ == "__main__":
    main()
