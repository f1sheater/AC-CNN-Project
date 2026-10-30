import os
import glob
import copy
import numpy as np
import wfdb
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = "data/mitdb"
RESULTS_DIR = "results"

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NORMAL = {"N"}
ABNORMAL = {"L", "R", "B", "A", "a", "J", "S",
            "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"}


def load_data():
    X, y, groups = [], [], []

    for hea in sorted(glob.glob(os.path.join(DATA_DIR, "*.hea"))):
        record_name = os.path.splitext(hea)[0]
        record_id = os.path.basename(record_name)

        patient_group = (
            "201_202"
            if record_id in {"201", "202"}
            else record_id
        )

        try:
            record = wfdb.rdrecord(record_name)
            ann = wfdb.rdann(record_name, "atr")
        except Exception as exc:
            print(f"Skipping {record_name}: {exc}")
            continue

        signal = record.p_signal[:, 0]
        fs = record.fs
        half_window = int(0.25 * fs)

        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in NORMAL and symbol not in ABNORMAL:
                continue

            start = sample - half_window
            end = sample + half_window

            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end].astype(np.float32)

            beat = (beat - beat.mean()) / (beat.std() + 1e-6)

            X.append(beat)
            y.append(0 if symbol in NORMAL else 1)
            groups.append(patient_group)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups)
    )


class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X[:, None, :]),
        torch.from_numpy(y).long()
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * len(yb)
            total_samples += len(yb)

    return total_loss / max(total_samples, 1)


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Device:", DEVICE)

    X, y, groups = load_data()

    print("Dataset:", X.shape)
    print(
        "Class counts (normal, abnormal):",
        np.bincount(y, minlength=2)
    )

    if len(X) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError(
            "No usable data or only one class was found."
        )

    records = np.unique(groups)

    trainval_records, test_records = train_test_split(
        records,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    train_records, val_records = train_test_split(
        trainval_records,
        test_size=0.125,
        random_state=RANDOM_SEED + 1
    )

    train_idx = np.flatnonzero(np.isin(groups, train_records))
    val_idx = np.flatnonzero(np.isin(groups, val_records))
    test_idx = np.flatnonzero(np.isin(groups, test_records))

    # Verify that no patient groups overlap
    assert set(train_records).isdisjoint(val_records)
    assert set(train_records).isdisjoint(test_records)
    assert set(val_records).isdisjoint(test_records)
    
    train_set = set(train_records)
    val_set = set(val_records)
    test_set = set(test_records)
    
    print("\nSubject independence check:")
    
    print("Training groups:", sorted(train_set))
    print("Validation groups:", sorted(val_set))
    print("Test groups:", sorted(test_set))
    
    print("Train/validation overlap:", train_set & val_set)
    print("Train/test overlap:", train_set & test_set)
    print("Validation/test overlap:", val_set & test_set)
    
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    print("SUCCESS: No subject groups overlap between partitions.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\nSplit summary:")
    for name, indices, labels, split_records in [
        ("Train", train_idx, y_train, train_records),
        ("Validation", val_idx, y_val, val_records),
        ("Test", test_idx, y_test, test_records),
    ]:
        print(
            f"{name}: {len(indices)} beats, "
            f"{len(split_records)} groups, "
            f"class counts={np.bincount(labels, minlength=2)}"
        )

    counts = np.bincount(y_train, minlength=2)
    weights = len(y_train) / (
        2.0 * np.maximum(counts, 1)
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    model = ECGCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    print("\nTraining model...")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(yb)
            total_samples += len(yb)

        train_loss = total_train_loss / max(total_samples, 1)
        val_loss = evaluate_loss(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a model.")

    model.load_state_dict(best_state)

    model_path = os.path.join(RESULTS_DIR, "ecg_cnn.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model to {model_path}")

    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            predictions.extend(
                logits.argmax(dim=1).cpu().numpy()
            )
            targets.extend(yb.numpy())

    print("\nFinal test classification report:")
    print(
        classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Abnormal"],
            zero_division=0
        )
    )

    print("Final test confusion matrix:")
    print(
        confusion_matrix(
            targets,
            predictions,
            labels=[0, 1]
        )
    )


if __name__ == "__main__":
    main()
