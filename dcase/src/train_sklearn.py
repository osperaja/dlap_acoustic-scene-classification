import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from models import SklearnAudioClassifier, SklearnAudioEnsembleClassifier
    from dataset import AcousticScenesDataset
except ImportError:
    from .models import SklearnAudioClassifier, SklearnAudioEnsembleClassifier
    from .dataset import AcousticScenesDataset

def build_final_estimator(cfg):
    if cfg["type"] == "logistic_regression":
        return LogisticRegression(**cfg.get("params", {}))
    raise ValueError(f"Unknown final_estimator type: {cfg['type']}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
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

def build_base_classifier(cfg, global_config):
    return SklearnAudioClassifier(
        classifier_type=cfg["name"],
        sample_rate=global_config["data"]["sample_rate"],
        n_mels=cfg.get("n_mels", 40),
        **cfg.get("params", {})
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print(f"Training: {config['model']}")
    print("=" * 50)

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

    if config["model"] == "SklearnAudioEnsembleClassifier":
        base_classifiers = config["ensemble"]["base_classifiers"]
        final_estimator = None
        if config["ensemble"]["type"] == "stacking":
            final_estimator = config["ensemble"]["final_estimator"]
        model = SklearnAudioEnsembleClassifier(
            base_classifiers=base_classifiers,
            final_estimator=final_estimator,
            ensemble_type=config["ensemble"]["type"],
        )
    else:
        model = SklearnAudioClassifier(
            classifier_type=config["network"]["type"],
            sample_rate=config["data"]["sample_rate"],
            n_mels=config["network"]["n_mels"],
            **config["network"].get("params", {})
        )

    print("\nModel:")
    print(model)
    if hasattr(model, "pipeline"):
        print("\nPipeline:")
        print(model.pipeline)
        print("\nPipeline params:")
        print(model.pipeline.get_params())
    elif hasattr(model, "get_params"):
        print("\nParams:")
        print(model.get_params())

    model.fit(dataloader=train_loader)

    print("Predicting training set:")
    train_acc = model.score(train_loader)

    print(f"\nTrain accuracy: {train_acc:.3f}")

    print("Predicting validation set:")
    val_acc = model.score(val_loader)
    val_pred, y_val = model.predict(val_loader)

    print(f"Validation accuracy: {val_acc:.3f}")

    print("\n" + "=" * 50)
    print("results:")
    print("=" * 50)
    print(f"val acc:   {val_acc:.3f}")

    # classification report
    class_names = sorted(train_dataset.meta_df['scene_name'].unique())
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=class_names))

    plot_confusion_matrix(y_val, val_pred, class_names,
                          save_path=config.get("logging", {}).get("cm_path"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model']}_{timestamp}-val_acc={val_acc:.2f}"
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.ckpt")

    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'val_accuracy': val_acc,
            'config': config,
        }, f)

if __name__ == "__main__":
    main()
