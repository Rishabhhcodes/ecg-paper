"""
CNN training loop.

Kept separate from run_grid.py so it can be imported in Colab without
dragging in the sklearn/xgboost path.
"""
import numpy as np


def train_cnn(Xtr, rrtr, ytr, Xva, rrva, yva, *, use_rr, strategy,
              epochs=60, batch=128, lr=1e-3, patience=12, min_epochs=20, seed=0,
              device=None, verbose=True, predict_extra=None):
    """Train one CNN and return validation probabilities.

    Early stopping is on validation macro-F1, not loss. With this level
    of imbalance, loss keeps improving while minority classes get worse,
    so stopping on loss selects the wrong epoch.

    predict_extra -- optional (X, rr) for a further set to score with the
    restored best-epoch model, returned as meta['proba_extra']. Added for
    Stage 4, where the model must be fitted on DS1 (with one fold held
    out for early stopping) and then applied to DS2, which must never be
    seen during fitting. Default None, so every existing call behaves
    exactly as before.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    import config as C
    from imbalance import class_weights, focal_loss_torch
    from models import make_cnn, count_params

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_cnn(use_rr, seed=seed).to(device)

    # Loss selection — exactly one strategy, never combined.
    if strategy == 'class_weight':
        w = torch.tensor(class_weights(ytr), device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    elif strategy == 'focal':
        alpha = torch.tensor(class_weights(ytr), device=device)
        criterion = lambda o, t: focal_loss_torch(o, t, gamma=2.0, alpha=alpha)  # noqa: E731
    else:                          # 'none' and 'adasyn' use plain CE;
        criterion = nn.CrossEntropyLoss()   # adasyn acts on the data instead

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5,
                                                       patience=3)

    to_t = lambda a, d=torch.float32: torch.tensor(a, dtype=d)  # noqa: E731
    tr = torch.utils.data.TensorDataset(to_t(Xtr), to_t(rrtr),
                                        to_t(ytr, torch.long))
    loader = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,
                                         drop_last=False)

    Xva_t = to_t(Xva).to(device)
    rrva_t = to_t(rrva).to(device)

    best_f1, best_state, best_epoch, stale = -1.0, None, -1, 0

    for ep in range(epochs):
        model.train()
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb, rb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = []
            for i in range(0, len(Xva_t), 2048):
                out = model(Xva_t[i:i + 2048], rrva_t[i:i + 2048])
                probs.append(torch.softmax(out, 1).cpu().numpy())
            proba = np.vstack(probs)

        # Early stopping and LR scheduling track macro-F1 over
        # EVAL_CLASSES only. Including Q would let its noisy, near-random
        # F1 (8 training beats in total) influence which epoch is selected.
        f1 = f1_score(yva, proba.argmax(1), average='macro',
                      labels=C.EVAL_LABELS, zero_division=0)
        sched.step(f1)

        if f1 > best_f1:
            best_f1, best_epoch, stale = f1, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_proba = proba
        else:
            stale += 1

        if verbose and (ep % 5 == 0 or stale == 0):
            print(f'      ep {ep:>3}  loss {loss.item():.4f}  '
                  f'val macroF1 {f1:.4f}{"  *" if stale == 0 else ""}')

        # Minimum epochs before stopping is allowed. Minority-class F1 sits
        # near zero for the first several epochs while the model learns the
        # easy majority, so a plain patience counter fires during that
        # plateau and the run ends around epoch 10 having never attempted
        # S or F. The whole 40-fold grid finishing in 20 minutes on a T4 is
        # the symptom.
        if ep >= min_epochs and stale >= patience:
            if verbose:
                print(f'      early stop at epoch {ep}')
            break

    model.load_state_dict(best_state)

    meta = {
        'best_epoch': int(best_epoch),
        'best_val_macro_f1': float(best_f1),
        'params': int(count_params(model)),
        'device': str(device),
    }

    # Scored after the best epoch's weights are restored, so this is the
    # same model the returned validation probabilities describe -- not
    # the last epoch's. Nothing here feeds back into training.
    if predict_extra is not None:
        Xex, rrex = predict_extra
        Xex_t, rrex_t = to_t(Xex).to(device), to_t(rrex).to(device)
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xex_t), 2048):
                out = model(Xex_t[i:i + 2048], rrex_t[i:i + 2048])
                chunks.append(torch.softmax(out, 1).cpu().numpy())
        meta['proba_extra'] = np.vstack(chunks)

    return best_proba, meta