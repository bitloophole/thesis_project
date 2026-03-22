from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LayerCache:
    x: np.ndarray
    z: np.ndarray
    a: np.ndarray


class NumpyMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        seed: int = 42,
        class_weights: np.ndarray | None = None,
        focal_gamma: float = 0.0,
        gradient_clip_norm: float | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.gradient_clip_norm = gradient_clip_norm
        layer_dims = (input_dim,) + hidden_dims + (output_dim,)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_params(layer_dims)

    def _init_params(self, layer_dims: tuple[int, ...]) -> None:
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=False):
            limit = np.sqrt(2.0 / in_dim)
            self.weights.append(self.rng.normal(0.0, limit, size=(in_dim, out_dim)).astype(np.float32))
            self.biases.append(np.zeros((1, out_dim), dtype=np.float32))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[LayerCache]]:
        activations = x
        caches: list[LayerCache] = []
        last_index = len(self.weights) - 1

        for idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=False)):
            z = activations @ w + b
            if idx == last_index:
                activations = softmax(z)
            else:
                activations = relu(z)
            caches.append(LayerCache(x=x if idx == 0 else caches[idx - 1].a, z=z, a=activations))
        return activations, caches

    def backward(self, x: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, caches: list[LayerCache]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_size = x.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        delta = y_prob.copy()
        true_probs = np.clip(y_prob[np.arange(batch_size), y_true], 1e-9, 1.0)
        delta[np.arange(batch_size), y_true] -= 1.0
        sample_weights = np.ones((batch_size, 1), dtype=np.float32)
        if self.class_weights is not None:
            sample_weights *= self.class_weights[y_true].reshape(-1, 1)
        if self.focal_gamma > 0.0:
            focal_weight = np.power(1.0 - true_probs, self.focal_gamma, dtype=np.float32).reshape(-1, 1)
            sample_weights *= focal_weight.astype(np.float32)
        delta *= sample_weights
        delta /= batch_size

        for layer_idx in reversed(range(len(self.weights))):
            prev_activation = x if layer_idx == 0 else caches[layer_idx - 1].a
            grad_w[layer_idx] = prev_activation.T @ delta + self.weight_decay * self.weights[layer_idx]
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                delta = delta @ self.weights[layer_idx].T
                delta = delta * relu_grad(caches[layer_idx - 1].z)

        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0.0:
            grad_w, grad_b = clip_gradients(grad_w, grad_b, self.gradient_clip_norm)

        return grad_w, grad_b

    def update(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * grad_w[idx]
            self.biases[idx] -= self.learning_rate * grad_b[idx]

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        patience: int | None = None,
        eval_every: int = 1,
        log_train_loss: bool = True,
    ) -> list[dict[str, float]]:
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")

        history: list[dict[str, float]] = []
        indices = np.arange(len(y_train))
        best_val_loss = float("inf")
        best_params: list[np.ndarray] | None = None
        patience_left = patience

        for epoch in range(1, epochs + 1):
            self.rng.shuffle(indices)
            x_epoch = x_train[indices]
            y_epoch = y_train[indices]

            for start in range(0, len(y_epoch), batch_size):
                end = start + batch_size
                x_batch = x_epoch[start:end]
                y_batch = y_epoch[start:end]
                y_prob, caches = self.forward(x_batch)
                grad_w, grad_b = self.backward(x_batch, y_batch, y_prob, caches)
                self.update(grad_w, grad_b)

            should_evaluate = (epoch % eval_every == 0) or (epoch == epochs)
            train_loss = None
            val_loss = None
            if should_evaluate:
                if log_train_loss:
                    train_loss = self.loss(x_train, y_train)
                val_loss = self.loss(x_val, y_val)

            history_entry = {"epoch": float(epoch)}
            if train_loss is not None:
                history_entry["train_loss"] = train_loss
            if val_loss is not None:
                history_entry["val_loss"] = val_loss
            history.append(history_entry)

            # Keep the best validation checkpoint to avoid drifting after the optimum.
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = self.get_parameters()
                if patience is not None:
                    patience_left = patience
            elif val_loss is not None and patience is not None:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_params is not None:
            self.set_parameters(best_params)
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_prob, _ = self.forward(x)
        return y_prob

    def loss(self, x: np.ndarray, y_true: np.ndarray) -> float:
        y_prob, _ = self.forward(x)
        clipped = np.clip(y_prob, 1e-9, 1.0)
        true_probs = clipped[np.arange(len(y_true)), y_true]
        log_likelihood = -np.log(true_probs)
        if self.class_weights is not None:
            log_likelihood = log_likelihood * self.class_weights[y_true]
        if self.focal_gamma > 0.0:
            log_likelihood = log_likelihood * np.power(1.0 - true_probs, self.focal_gamma)
        ce_loss = np.mean(log_likelihood)
        reg_loss = 0.5 * self.weight_decay * sum(np.sum(w * w) for w in self.weights)
        return float(ce_loss + reg_loss)

    def get_parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = []
        for w, b in zip(self.weights, self.biases, strict=False):
            params.append(w.copy())
            params.append(b.copy())
        return params

    def set_parameters(self, params: list[np.ndarray]) -> None:
        if len(params) != 2 * len(self.weights):
            raise ValueError("Parameter list length does not match the model")
        for idx in range(len(self.weights)):
            self.weights[idx] = params[2 * idx].copy()
            self.biases[idx] = params[2 * idx + 1].copy()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def clip_gradients(
    grad_w: list[np.ndarray],
    grad_b: list[np.ndarray],
    clip_norm: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    total_norm_sq = 0.0
    for grad in [*grad_w, *grad_b]:
        total_norm_sq += float(np.sum(grad * grad))
    total_norm = np.sqrt(total_norm_sq)
    if total_norm <= clip_norm or total_norm == 0.0:
        return grad_w, grad_b

    scale = clip_norm / (total_norm + 1e-12)
    clipped_w = [grad * scale for grad in grad_w]
    clipped_b = [grad * scale for grad in grad_b]
    return clipped_w, clipped_b
