import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def targeted_fgsm_attack(model, X, y, epsilon, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach()
    X_adv.requires_grad = True
    outputs = model(X_adv)
    criterion = nn.CrossEntropyLoss()
    y_tensor = torch.tensor(y, dtype=torch.long)
    loss = criterion(outputs, y_tensor)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * X_adv.grad.data.sign()

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
        perturbation = perturbation * importance_tensor.expand_as(perturbation)
        scale_factor = epsilon / torch.max(torch.abs(perturbation))
        perturbation = perturbation * scale_factor

    X_adv = X_adv + perturbation
    return X_adv.detach()

def targeted_pgd_attack(model, X, y, epsilon, alpha=0.01, num_iter=40, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach() + torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        criterion = nn.CrossEntropyLoss()
        y_tensor = torch.tensor(y, dtype=torch.long)
        loss = criterion(outputs, y_tensor)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = X_adv.grad * importance_tensor.expand_as(X_adv)
            if torch.max(torch.abs(grad)) > 0:
                grad = grad * alpha / torch.max(torch.abs(grad))
            X_adv = X_adv + grad.sign()
            eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
            eta_tensor = torch.tensor(eta, dtype=torch.float32)
            X_adv = torch.clamp(X_tensor + eta_tensor, min=X.min(), max=X.max())

    return X_adv.detach()

def targeted_bim_attack(model, X, y, epsilon, alpha=0.005, num_iter=10, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach()

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        criterion = nn.CrossEntropyLoss()
        y_tensor = torch.tensor(y, dtype=torch.long)
        loss = criterion(outputs, y_tensor)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = X_adv.grad * importance_tensor.expand_as(X_adv)
            if torch.max(torch.abs(grad)) > 0:
                grad = grad * alpha / torch.max(torch.abs(grad))
            X_adv = X_adv + grad.sign()
            eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
            eta_tensor = torch.tensor(eta, dtype=torch.float32)
            X_adv = torch.clamp(X_tensor + eta_tensor, min=X.min(), max=X.max())

    return X_adv.detach()

def targeted_transfer_attack(surrogate_model, X, y, epsilon, feature_importance=None):
    return targeted_fgsm_attack(surrogate_model, X, y, epsilon, feature_importance)

def targeted_random_attack(model, X, y, epsilon, num_trials, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    best_X_adv = X_tensor.clone().detach()

    best_acc = 1.0

    if feature_importance is not None:
        importance = feature_importance / np.sum(feature_importance) * feature_importance.shape[0]
        importance_tensor = torch.tensor(importance, dtype=torch.float32).view(1, -1)
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_trials):
        base_noise = torch.empty_like(X).uniform_(-1, 1)
        noise = base_noise * importance_tensor.expand_as(base_noise) * epsilon
        if torch.max(torch.abs(noise)) > epsilon:
            noise = noise * epsilon / torch.max(torch.abs(noise))

        X_candidate = torch.clamp(X + noise, min=X.min(), max=X.max())

        if isinstance(model, nn.Module):
            outputs = model(X_candidate)
            _, predicted = torch.max(outputs, 1)
            acc_candidate = (predicted == y).float().mean().item()
        else:  # Для моделей sklearn
            X_np = X_candidate.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
            y_pred = model.predict(X_np)
            acc_candidate = accuracy_score(y_np, y_pred)

        if acc_candidate < best_acc:
            best_acc = acc_candidate
            best_X_adv = X_candidate

    return best_X_adv.detach()

def targeted_square_attack(model, X, y, epsilon=0.3, iters=100, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach()

    if feature_importance is not None:
        importance = feature_importance / np.sum(feature_importance)
        feature_probs = importance / np.sum(importance)
    else:
        feature_probs = np.ones(X.shape[1]) / X.shape[1]

    for _ in range(iters):
        selected_features = np.random.choice(
            X.shape[1],
            size=max(1, int(X.shape[1] * 0.1)),
            replace=False,
            p=feature_probs
        )
        mask = torch.zeros_like(X_adv)
        mask[:, selected_features] = 1.0
        perturbation = (torch.rand_like(X_adv) - 0.5) * 2 * epsilon
        X_adv = X_adv + mask * perturbation
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        eta_tensor = torch.tensor(eta, dtype=torch.float32)
        X_adv = torch.clamp(X_tensor + eta_tensor, min=X.min(), max=X.max())

    return X_adv.detach()

def targeted_fgsm_attack_boost(model, X, y, epsilon, feature_importance=None):
    # Создаем случайное возмущение
    perturbation = torch.randn_like(X) * epsilon

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
        perturbation = perturbation * importance_tensor.expand_as(perturbation)
        scale_factor = epsilon / torch.max(torch.abs(perturbation))
        perturbation = perturbation * scale_factor

    # Применяем возмущение
    X_adv = X + perturbation
    return X_adv.detach()

def targeted_pgd_attack_boost(model, X, y, epsilon, alpha=0.01, num_iter=40, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach() + torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_iter):
        # Создаем случайное возмущение
        perturbation = torch.randn_like(X_adv) * alpha

        if feature_importance is not None:
            perturbation = perturbation * importance_tensor.expand_as(perturbation)

        # Применяем возмущение
        X_adv = X_adv + perturbation
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        eta_tensor = torch.tensor(eta, dtype=torch.float32)
        X_adv = torch.clamp(X_tensor + eta_tensor, min=X.min(), max=X.max())

    return X_adv.detach()

def targeted_bim_attack_boost(model, X, y, epsilon, alpha=0.005, num_iter=10, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_adv = X_tensor.clone().detach()

    if feature_importance is not None:
        importance_tensor = torch.tensor(feature_importance, dtype=torch.float32).view(1, -1)
        importance_tensor = importance_tensor / importance_tensor.sum()
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_iter):
        # Создаем случайное возмущение
        perturbation = torch.randn_like(X_adv) * alpha

        if feature_importance is not None:
            perturbation = perturbation * importance_tensor.expand_as(perturbation)

        # Применяем возмущение
        X_adv = X_adv + perturbation
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        eta_tensor = torch.tensor(eta, dtype=torch.float32)
        X_adv = torch.clamp(X_tensor + eta_tensor, min=X.min(), max=X.max())

    return X_adv.detach()

def targeted_random_attack_boost(model, X, y, epsilon, num_trials, feature_importance=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    best_X_adv = X_tensor.clone().detach()
    best_acc = 1.0

    if feature_importance is not None:
        importance = feature_importance / np.sum(feature_importance) * feature_importance.shape[0]
        importance_tensor = torch.tensor(importance, dtype=torch.float32).view(1, -1)
    else:
        importance_tensor = torch.ones((1, X.shape[1]), dtype=torch.float32)

    for _ in range(num_trials):
        base_noise = torch.empty_like(X).uniform_(-1, 1)
        noise = base_noise * importance_tensor.expand_as(base_noise) * epsilon
        if torch.max(torch.abs(noise)) > epsilon:
            noise = noise * epsilon / torch.max(torch.abs(noise))

        X_candidate = torch.clamp(X + noise, min=X.min(), max=X.max())

        X_candidate_np = X_candidate.detach().cpu().numpy()
        y_pred = model.predict(X_candidate_np)
        acc_candidate = accuracy_score(y.detach().cpu().numpy(), y_pred)

        if acc_candidate < best_acc:
            best_acc = acc_candidate
            best_X_adv = X_candidate

    return best_X_adv.detach()

def evaluate_attack_with_epsilon(model, data_loader, attack_func, epsilon_range, X_test_tensor, y_test_tensor, feature_importance=None):
    results = []
    for epsilon in tqdm(epsilon_range, desc="Testing epsilon values"):
        x_test_batch = X_test_tensor[:100]
        y_test_batch = y_test_tensor[:100]
        x_test_adv = attack_func(model, x_test_batch, y_test_batch, epsilon, feature_importance)
        model.eval()
        with torch.no_grad():
            outputs = model(x_test_adv)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_batch).float().mean().item()
        results.append((epsilon, accuracy))
    return results

def evaluate_attack(model, adv_X, y_test_tensor):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            outputs = model(adv_X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_tensor[:len(adv_X)]).float().mean().item()
        return accuracy
    else:  # Для модели бустинга
        adv_X_np = adv_X.cpu().numpy()
        y_pred = model.predict(adv_X_np)
        return accuracy_score(y_test[:len(adv_X_np)], y_pred)
