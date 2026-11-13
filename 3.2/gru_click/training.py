"""Measured model selection, resumable training, and bounded wall-clock budgets."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from .common import (
    atomic_json,
    cpu_tree,
    restore_rng,
    rng_state,
    save_torch,
    seed_all,
)
from .model import GRUProductModel


@torch.inference_mode()
def infer(model, store, rows, device, context, batch_size=128, embeddings=False):
    model.eval()
    scores, users, candidates = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = store.batch(rows[start : start + batch_size], context, device)
        if embeddings:
            s, u, c = model(**batch, return_features=True)
            users.append(u.cpu().numpy())
            candidates.append(c.cpu().numpy())
        else:
            s = model(**batch)
        if not torch.isfinite(s).all():
            raise ValueError("Nonfinite predictions")
        scores.append(s.cpu().numpy())
    result = dict(scores=np.concatenate(scores))
    if embeddings:
        result.update(
            users=np.concatenate(users), candidates=np.concatenate(candidates)
        )
    return result


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = GRUProductModel(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def fit_gru(
    store,
    rows,
    validation,
    config,
    destination,
    *,
    device,
    context=128,
    batch_size=64,
    epochs=4,
    learning_rate=0.001,
    deadline,
    initial=None,
    log=print,
    update=lambda **kw: None,
    clock=time.time,
    stop_after_steps=None,
):
    destination = Path(destination)
    best_path, resume = (
        destination.with_suffix(".pt"),
        destination.with_suffix(".resume.pt"),
    )
    seed_all()
    model = GRUProductModel(**config).to(device)
    if initial is not None:
        model.load_state_dict(
            torch.load(initial, map_location="cpu", weights_only=True)["state_dict"]
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    settings = dict(
        config=config,
        context=context,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        deadline=deadline,
        validation=validation is not None,
        rows_sha256=__import__("hashlib")
        .sha256(np.asarray(rows, dtype=np.int64).tobytes())
        .hexdigest(),
    )
    state = dict(
        epoch=0,
        group=0,
        steps=0,
        loss_sum=0.0,
        mass=0.0,
        best_accuracy=-1.0,
        best_epoch=0,
        curve=[],
        finished=False,
        deadline_reached=False,
    )
    coverage = np.zeros(store.n_original, bool)
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["settings"] != settings:
            raise ValueError("Resume arguments changed; use a new output directory")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        coverage = saved["coverage"].numpy().copy()
        restore_rng(saved["rng"], device)
        del saved

    def checkpoint():
        save_torch(
            resume,
            dict(
                settings=settings,
                state=state.copy(),
                coverage=torch.from_numpy(coverage.copy()),
                state_dict=cpu_tree(model.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()),
                rng=rng_state(device),
            ),
        )

    if state["finished"]:
        return load_model(best_path, device), state
    groups = math.ceil(len(rows) / batch_size)
    called_steps = 0
    for epoch in range(state["epoch"], epochs):
        model.train()
        order = np.random.default_rng(42 + epoch).permutation(rows)
        timed_out = False
        for group in range(state["group"], groups):
            if clock() >= deadline:
                timed_out = True
                state["deadline_reached"] = True
                break
            selected = order[group * batch_size : (group + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            absolute = epoch * groups + group
            factor = min(1.0, (absolute + 1) / min(100, groups)) * (
                0.2
                + 0.8
                * 0.5
                * (1 + math.cos(math.pi * absolute / max(1, epochs * groups - 1)))
            )
            for p in optimizer.param_groups:
                p["lr"] = learning_rate * factor
            batch = store.batch(selected, context, device)
            weights = torch.as_tensor(store.index["weights"][selected], device=device)
            labels = torch.as_tensor(store.labels[selected], device=device)
            loss = (
                torch.nn.functional.cross_entropy(
                    model(**batch), labels, reduction="none", label_smoothing=0.01
                )
                * weights
            ).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mass = float(weights.sum().cpu())
            value = float(loss.detach().cpu())
            state.update(
                epoch=epoch,
                group=group + 1,
                steps=state["steps"] + 1,
                loss_sum=state["loss_sum"] + value * mass,
                mass=state["mass"] + mass,
            )
            coverage[selected[selected < store.n_original]] = True
            called_steps += 1
            if (group + 1) % 25 == 0 or group + 1 == groups:
                update(
                    epoch=epoch + 1,
                    epochs=epochs,
                    group=group + 1,
                    groups=groups,
                    loss=state["loss_sum"] / state["mass"],
                )
                log(
                    f"{destination.name}: epoch {epoch + 1}/{epochs}, step {group + 1}/{groups}, loss={state['loss_sum'] / state['mass']:.4f}"
                )
            if (group + 1) % 100 == 0 or (
                stop_after_steps is not None and called_steps >= stop_after_steps
            ):
                checkpoint()
            if stop_after_steps is not None and called_steps >= stop_after_steps:
                return model, state
        accuracy = None
        if validation is not None:
            predictions = infer(model, store, validation, device, context)[
                "scores"
            ].argmax(1)
            accuracy = float(np.mean(predictions == store.labels[validation]))
        info = dict(
            epoch=epoch + 1,
            partial=timed_out,
            early_stop_accuracy=accuracy,
            original_endpoints_seen=int(coverage.sum()),
            train_loss=state["loss_sum"] / max(1, state["mass"]),
        )
        if validation is None or accuracy > state["best_accuracy"]:
            if accuracy is not None:
                state["best_accuracy"] = accuracy
            state["best_epoch"] = epoch + 1
            save_torch(
                best_path,
                dict(
                    config=config,
                    state_dict=cpu_tree(model.state_dict()),
                    selection=info,
                    context=context,
                ),
            )
        state["curve"].append(info)
        early_stop = validation is not None and epoch + 1 - state["best_epoch"] >= 2
        state.update(
            epoch=epoch + 1,
            group=0,
            loss_sum=0.0,
            mass=0.0,
            finished=timed_out or early_stop or epoch + 1 == epochs,
        )
        checkpoint()
        atomic_json(destination.with_suffix(".json"), state)
        log(
            f"{destination.name}: saved epoch {epoch + 1}, early-stop accuracy={accuracy}, partial={timed_out}"
        )
        if state["finished"]:
            break
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"]
    )
    return model, state
