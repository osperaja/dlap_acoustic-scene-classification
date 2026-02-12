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
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=None,
                          min_confusion=0.10):
    """
    Normalised confusion matrix showing only classes with max misclassification >= min_confusion.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Only keep rows/cols where max misclassification >= threshold
    # Diagonal counts as correct, so we check off-diagonal max
    off_diag_max = cm_normalized.copy()
    np.fill_diagonal(off_diag_max, 0)
    keep_rows = np.where(off_diag_max.max(axis=1) >= min_confusion)[0]
    keep_cols = np.where(off_diag_max.max(axis=0) >= min_confusion)[0]

    # filter cm and class names
    cm_filtered = cm_normalized[np.ix_(keep_rows, keep_cols)]
    class_names_filtered = [class_names[i] for i in keep_rows]

    # plot
    fig, ax = plt.subplots(figsize=(len(class_names_filtered), len(class_names_filtered)))
    im = ax.imshow(cm_filtered, interpolation='nearest', cmap=plt.cm.Blues)
    # ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names_filtered)),
           yticks=np.arange(len(class_names_filtered)),
           xticklabels=class_names_filtered,
           yticklabels=class_names_filtered,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm_filtered.max() / 2.
    for i in range(cm_filtered.shape[0]):
        for j in range(cm_filtered.shape[1]):
            if cm_filtered[i, j] >= min_confusion:
                ax.text(j, i, format(cm_filtered[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_filtered[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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

    plot_confusion_matrix(y_val, val_pred, class_names, save_path=config.get("logging", {}).get("cm_path"))
    plot_confusion_matrix_thresh(y_val, val_pred, class_names, save_path=config.get("logging", {}).get("cm_path"))

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
