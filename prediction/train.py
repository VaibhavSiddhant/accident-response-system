"""
Training Script — FT-Transformer for India Accident Risk Prediction
Run: python train.py --epochs 100 --data data/accident_prediction_india.csv
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.preprocess import prepare_pipeline, generate_synthetic_data, encode_and_scale
from models.ft_transformer import build_model


def make_loaders(X_train, y_train, X_val, y_val, batch_size=256, balance=True):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    if balance:
        class_counts = np.bincount(y_train)
        weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size*2, shuffle=False)
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss, total = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        total += len(y_batch)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    return {
        'loss':     total_loss / total,
        'accuracy': (all_preds == all_labels).mean(),
        'f1':       f1_score(all_labels, all_preds, average='macro'),
        'auc':      roc_auc_score(all_labels, all_probs),
        'preds':    all_preds,
        'probs':    all_probs,
        'labels':   all_labels,
    }


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*55}")
    print(f"  India Accident Risk Predictor — FT-Transformer")
    print(f"  Device: {device.upper()}")
    print(f"{'='*55}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        print(f"Loading real dataset: {args.data}")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_pipeline(args.data)
    else:
        print("No dataset found — using synthetic data.\n")
        df = generate_synthetic_data(n_samples=10000)
        df['target'] = (df['severity'].isin(['Fatal', 'Serious'])).astype(int)
        print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
        X, y = encode_and_scale(df, fit=True)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        print(f"Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = X_train.shape[1]
    model = build_model(n_features, size=args.model_size, device=device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Train ─────────────────────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    print(f"Training for {args.epochs} epochs (patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} F1: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'model_size': args.model_size,
                'val_f1': best_f1,
                'val_auc': val_metrics['auc'],
            }, 'checkpoints/best_model.pt')
            print(f"  ✓ Saved best model (val F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Final Test Set Evaluation")
    print(f"{'='*55}")

    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Best Val F1: {best_f1:.4f}")
    print(f"\nFinal Classification Report:")
    print(classification_report(
        test_metrics['labels'], test_metrics['preds'],
        target_names=['Minor', 'Serious/Fatal']
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        type=str,   default=None)
    parser.add_argument('--model_size',  type=str,   default='medium', choices=['small','medium','large'])
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=15)
    args = parser.parse_args()
    train(args)