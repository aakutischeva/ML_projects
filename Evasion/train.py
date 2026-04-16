import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from utils import log_print
from attacks import (
    targeted_fgsm_attack,
    targeted_pgd_attack,
    targeted_bim_attack
)
import warnings
warnings.filterwarnings("ignore")

# Функция для вывода важности признаков в виде таблицы
def print_feature_importance_prettytable(top_features, feature_importance):
    from prettytable import PrettyTable
    importance_values = feature_importance[top_features]
    table = PrettyTable()
    table.field_names = ["№ п/п", "№ признака", "Важность", "% от общей"]
    for i, (idx, importance) in enumerate(zip(top_features, importance_values), 1):
        percent = importance / feature_importance.sum() * 100
        table.add_row([i, idx, f"{importance:.5f}", f"{percent:.2f}%"])
    table.align = "r"
    table.title = "ВАЖНЫЕ ПРИЗНАКИ МОДЕЛИ"
    log_print(table)
    total_importance = importance_values.sum()
    percent = total_importance / feature_importance.sum() * 100
    log_print(f"\nСуммарная важность выбранных признаков: {total_importance:.5f}")
    log_print(f"Процент от общей важности всех признаков: {percent:.2f}%")

# Функция для обучения суррогатной модели
def train_surrogate(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        log_print(f"Surrogate Epoch {epoch + 1}/{num_epochs}")

# Функция для обучения очисителя
def train_purifier(purifier, train_loader, purifier_optimizer, purifier_criterion, num_epochs_purifier, model, epsilon, importance_mask):
    for epoch in range(num_epochs_purifier):
        for X_batch, y_batch in train_loader:
            X_batch_adv = targeted_fgsm_attack(model, X_batch, y_batch, epsilon=epsilon, feature_importance=importance_mask)
            purifier_optimizer.zero_grad()
            purified_clean = purifier(X_batch)
            purified_adv = purifier(X_batch_adv)
            loss_clean = purifier_criterion(purified_clean, X_batch)
            loss_adv = purifier_criterion(purified_adv, X_batch)
            loss = loss_clean + 2.0 * loss_adv
            loss.backward()
            purifier_optimizer.step()
        log_print(f"Purifier Epoch {epoch + 1}/{num_epochs_purifier}, Clean Loss: {loss_clean.item():.6f}, Adv Loss: {loss_adv.item():.6f}")

# Функция для генерации смешанных адверсариальных примеров
def mixed_adversarial_examples(model, purifier, X, y, epsilon=0.1, importance_mask=None):
    batch_size = X.shape[0]
    split1, split2 = batch_size // 3, 2 * (batch_size // 3)
    X_fgsm = targeted_fgsm_attack(model, X[:split1], y[:split1], epsilon, importance_mask)
    X_pgd = targeted_pgd_attack(model, X[split1:split2], y[split1:split2], epsilon, alpha=0.01, num_iter=5, feature_importance=importance_mask)
    X_bim = targeted_bim_attack(model, X[split2:], y[split2:], epsilon, alpha=0.005, num_iter=3, feature_importance=importance_mask)
    X_adv = torch.cat([X_fgsm, X_pgd, X_bim], dim=0)
    return X_adv

# Функция для обучения модели с адверсариальным обучением
def adversarial_training(model, train_loader, criterion, optimizer, num_epochs, attack_fn, purifier=None, epsilon=0.1):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch_adv = attack_fn(model, purifier, X_batch, y_batch, epsilon)
            X_batch_adv_purified = purifier(X_batch_adv)
            outputs_clean = model(X_batch)
            loss_clean = criterion(outputs_clean, y_batch)
            outputs_adv = model(X_batch_adv)
            loss_adv = criterion(outputs_adv, y_batch)
            outputs_purified = model(X_batch_adv_purified)
            loss_purified = criterion(outputs_purified, y_batch)
            loss = loss_clean + 0.5 * loss_adv + loss_purified
            loss.backward()
            optimizer.step()
        log_print(f"Adversarial Training Epoch {epoch + 1}/{num_epochs}")

# Функция для обучения модели бустинга с адверсариальными примерами
def train_boosting_with_adversarial_examples(boost_model, model, purifier, X_train, y_train, attack_fn, epsilon=0.1, batch_size=100):
    adv_examples = []
    adv_labels = []

    for i in range(0, len(X_train), batch_size):
        end = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end]
        y_batch = y_train[i:end]
        y_batch_tensor = torch.tensor(y_batch)
        # Генерация адверсариальных примеров с разными атаками
        X_adv = attack_fn(model, purifier, X_batch, y_batch, epsilon)
        adv_examples.append(X_adv.detach().cpu().numpy())
        adv_labels.append(y_batch_tensor.cpu().numpy())

    # Объединение обычных и адверсариальных примеров
    X_train_augmented = np.vstack([X_train, np.vstack(adv_examples)])
    y_train_augmented = np.concatenate([y_train, np.concatenate(adv_labels)])

    boost_model.fit(X_train_augmented, y_train_augmented)

# Функция для создания очищенного набора данных
def create_purified_dataset(X_train_tensor, purifier, batch_size=100):
    X_train_purified = []
    for i in range(0, len(X_train_tensor), batch_size):
        end = min(i + batch_size, len(X_train_tensor))
        X_batch = X_train_tensor[i:end]
        X_purified = purifier(X_batch).detach().cpu().numpy()
        X_train_purified.append(X_purified)
    return np.vstack(X_train_purified)

# Функция для обучения модели с очищенными данными
def train_model_with_purified_data(model, X_train_purified, y_train, criterion, optimizer, num_epochs):
    X_train_purified_tensor = torch.tensor(X_train_purified, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_loader_purified = DataLoader(TensorDataset(X_train_purified_tensor, y_train_tensor), batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader_purified:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        log_print(f"Purified Model Epoch {epoch + 1}/{num_epochs}")

# Функция для очистки входных данных
def apply_defense(model, purifier, X_adv):
    X_purified = purifier(X_adv)
    model.eval()
    with torch.no_grad():
        outputs = model(X_purified)
        _, predicted = torch.max(outputs, 1)
    return predicted

# Оценка моделей с очисткой данных
def evaluate_with_purifier(model, purifier, adv_X, y, y_test):
    if isinstance(model, nn.Module):
        model.eval()
        purifier.eval()
        with torch.no_grad():
            purified_X = purifier(adv_X)
            outputs = model(purified_X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y[:len(adv_X)]).float().mean().item()
        return accuracy
    else:  # Для модели бустинга
        purifier.eval()
        with torch.no_grad():
            purified_X = purifier(adv_X).detach().cpu().numpy()
            y_pred = model.predict(purified_X)
            return accuracy_score(y_test[:len(purified_X)], y_pred)
