import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# Функция для оценки модели
def evaluate(model, X_test, y_test, data_loader=None, X=None, y=None):
    if isinstance(model, nn.Module):
        model.eval()
        if data_loader is not None:
            y_pred, y_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in data_loader:
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    y_pred.extend(predicted.cpu().numpy())
                    y_true.extend(y_batch.cpu().numpy())
            return accuracy_score(y_true, y_pred)
        else:
            with torch.no_grad():
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                return accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
    else:  # Для моделей sklearn
        if X is None and y is None:
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
        else:
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            y_pred = model.predict(X)
            return accuracy_score(y, y_pred)

# Функция для построения графиков зависимости точности от epsilon
def plot_epsilon_results(results, attack_name, model_name):
    epsilons = [res[0] for res in results]
    accuracies = [res[1] * 100 for res in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, accuracies, marker='o')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy vs Epsilon for {attack_name} Attack ({model_name})')
    ax.grid(True)

    save_plot(fig, f"epsilon_vs_accuracy_{attack_name}_{model_name}.png")

def save_plot(fig, filename):
    """Сохраняет график в папку plots"""
    path = os.path.join("plots", filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# Оценка атак на модель бустинга с разными значениями epsilon
def evaluate_boost_attack_with_epsilon(model, attack_func, epsilon_range, X_test_tensor, y_test_tensor, y_test, feature_importance=None):
    results = []
    for epsilon in tqdm(epsilon_range, desc="Testing epsilon values"):
        x_test_batch = X_test_tensor[:100]
        y_test_batch = y_test_tensor[:100]
        x_test_adv = attack_func(model, x_test_batch, y_test_batch, epsilon, feature_importance)
        x_test_adv_np = x_test_adv.detach().cpu().numpy()
        y_pred = model.predict(x_test_adv_np)
        accuracy = accuracy_score(y_test[:len(x_test_adv_np)], y_pred)
        results.append((epsilon, accuracy))
    return results

# Оценка атаки
def evaluate_attack(model, X_adv, X_test, y_test, y_true_tensor):
    adv_loader = DataLoader(TensorDataset(X_adv, y_true_tensor), batch_size=32, shuffle=False)
    return evaluate(model, X_test, y_test, adv_loader)

# Функция для построения матрицы ошибок
def plot_confusion_matrix(model, X, y, title, purifier_obj=None):
    if isinstance(model, nn.Module):  # Для нейронной сети
        model.eval()
        if purifier_obj is not None:
            with torch.no_grad():
                X = purifier_obj(X)
        y_pred = []
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
    else:  # Для модели бустинга
        if purifier_obj is not None:
            with torch.no_grad():
                X = purifier_obj(X).detach().cpu().numpy()
        else:
            X = X.detach().cpu().numpy()
        y_pred = model.predict(X)

    y_true = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    save_plot(fig, f"confusion_matrix_{title.replace(' ', '_')}.png")
