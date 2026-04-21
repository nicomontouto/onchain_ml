"""
model_lstm.py — LSTM para clasificacion binaria con PyTorch.

V2: train_lstm acepta lookback configurable y target_col.

Puede correrse de forma independiente:
  python model_lstm.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from feature_engineering import get_feature_cols

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MODEL_PATH = Path(__file__).parent / "best_lstm.pt"

LOOKBACK = 14
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15
GRAD_CLIP = 1.0


# --- Arquitectura ---

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# --- Construccion de secuencias ---

def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    """
    Crea secuencias de longitud lookback sin data leakage.
    X_seq[i] = features de los periodos [i, i+lookback)
    y_seq[i] = target del periodo i+lookback
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i: i + lookback])
        y_seq.append(y[i + lookback])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# --- Entrenamiento ---

def train_lstm(df: pd.DataFrame, lookback: int = LOOKBACK, target_col: str = "target") -> tuple:
    """
    Entrena el LSTM con early stopping y gradient clipping.

    V2: Acepta lookback configurable y target_col.

    Retorna:
      model, scaler, imputer, train_history (list de dicts), (X_te_seq, y_te_seq, all_proba)
    """
    print(f"\n[LSTM] Preparando datos... (lookback={lookback}, target={target_col})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [LSTM] Device: {device}")

    feature_cols = get_feature_cols(df)

    date_col = "datetime" if "datetime" in df.columns else "date"
    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"  [LSTM] Train: {len(train_df)} | Test: {len(test_df)}")

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train, lookback)
    X_te_seq, y_te_seq = make_sequences(X_test_sc, y_test, lookback)

    print(f"  [LSTM] Secuencias train: {X_tr_seq.shape} | test: {X_te_seq.shape}")

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        print("  [LSTM] Insuficientes secuencias. Retornando vacio.")
        return None, scaler, imputer, [], (X_te_seq, y_te_seq, np.array([]))

    train_ds = TensorDataset(
        torch.from_numpy(X_tr_seq),
        torch.from_numpy(y_tr_seq),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_te_seq),
        torch.from_numpy(y_te_seq),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_tr_seq.shape[2]
    model = LSTMClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\n[LSTM] Entrenando (max {MAX_EPOCHS} epochs, patience={PATIENCE})...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item() * len(Xb)
                correct += ((pred >= 0.5).float() == yb).sum().item()
        val_loss /= len(test_ds)
        val_acc = correct / len(test_ds)

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [LSTM] Early stopping en epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"  [LSTM] Mejor val_loss={best_val_loss:.4f} | guardado en {MODEL_PATH}")

    model.eval()
    all_proba = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            all_proba.extend(model(Xb.to(device)).cpu().numpy())

    all_proba = np.array(all_proba)
    all_pred = (all_proba >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_te_seq, all_pred)
    try:
        auc = roc_auc_score(y_te_seq, all_proba)
    except ValueError:
        auc = float("nan")
    print(f"  [LSTM] Final Test Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    return model, scaler, imputer, history, (X_te_seq, y_te_seq, all_proba)


# --- Entry point ---

if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Primero corre feature_engineering.py para generar features.parquet")
        exit(1)

    print("=" * 60)
    print("TEST INDEPENDIENTE: model_lstm.py v2")
    print("=" * 60)

    df = pd.read_parquet(features_path)
    print(f"Dataset cargado: {df.shape}")

    model, scaler, imputer, history, test_data = train_lstm(df, lookback=LOOKBACK)

    df_history = pd.DataFrame(history)
    df_history.to_parquet(PROCESSED_DIR / "lstm_history.parquet", index=False)
    print("\n[OK] model_lstm.py v2 corrio sin errores.")
