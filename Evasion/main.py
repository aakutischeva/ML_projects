import sys
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from model import MLP, SurrogateMLP, FeaturePurifier
from train import  (train_surrogate, adversarial_training, train_purifier, mixed_adversarial_examples, adversarial_training,
                    train_boosting_with_adversarial_examples, create_purified_dataset, train_model_with_purified_data,
                    apply_defense, evaluate_with_purifier, print_feature_importance_prettytable)
from evaluation import evaluate, evaluate_attack, plot_confusion_matrix,evaluate_boost_attack_with_epsilon,plot_epsilon_results
from torch.utils.data import DataLoader, TensorDataset
from utils import log_print
from attacks import (
    targeted_fgsm_attack,
    targeted_pgd_attack,
    targeted_bim_attack,
    targeted_transfer_attack,
    targeted_random_attack,
    targeted_square_attack,
    evaluate_attack_with_epsilon,
    targeted_random_attack_boost,
    targeted_bim_attack_boost,
    targeted_pgd_attack_boost,
    targeted_fgsm_attack_boost
)
import warnings
warnings.filterwarnings("ignore")


# ===========================
# === НАСТРОЙКА ЛОГГЕРА ===
# ===========================
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =========================================
# ===== 1. ЗАГРУЗКА ДАННЫХ И ОБУЧЕНИЕ =====
# =========================================
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

num_epochs = 50
num_epochs_sur = 50
num_epochs_def = 50
num_epochs_purifier = 50

# Создание модели бустинга
boost_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
boost_model.fit(X_train, y_train)

feature_importance = boost_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
top_features = sorted_idx[:int(len(sorted_idx) * 0.3)]
# Создание нейронной сети

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
model = MLP(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    log_print(f"Epoch {epoch + 1}/{num_epochs}")

acc_nn_before_attack = evaluate(model, X_test,y_test, test_loader)
acc_boost_before_attack = boost_model.score(X_test, y_test)
print_feature_importance_prettytable(top_features, feature_importance)
log_print(f"Точность нейронной сети до атаки: {acc_nn_before_attack * 100:.2f}%")
log_print(f"Точность модели бустинга до атаки: {acc_boost_before_attack * 100:.2f}%")

importance_mask = np.ones(X_train.shape[1])
importance_mask[top_features] = 3.0
importance_mask = importance_mask / importance_mask.sum() * len(importance_mask)

surrogate_model = SurrogateMLP(input_dim, num_classes)
criterion_sur = nn.CrossEntropyLoss()
optimizer_sur = optim.Adam(surrogate_model.parameters(), lr=0.001)

train_surrogate(surrogate_model, train_loader, criterion, optimizer, num_epochs_sur)

epsilon_values = np.linspace(0.01, 0.5, 100)

fgsm_results = evaluate_attack_with_epsilon(
    model,
    test_loader,
    targeted_fgsm_attack,
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    importance_mask
)
pgd_results = evaluate_attack_with_epsilon(
    model,
    test_loader,
    lambda m, x, y, e, i: targeted_pgd_attack(m, x, y, e, alpha=0.01, num_iter=10, feature_importance=i),
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    importance_mask
)
bim_results = evaluate_attack_with_epsilon(
    model,
    test_loader,
    lambda m, x, y, e, i: targeted_bim_attack(m, x, y, e, alpha=0.005, num_iter=5, feature_importance=i),
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    importance_mask
)
random_results = evaluate_attack_with_epsilon(
    model,
    test_loader,
    lambda m, x, y, e, i: targeted_random_attack(m, x, y, e, num_trials=50, feature_importance=i),
    epsilon_values,
    X_train_tensor,
    y_train_tensor,
    importance_mask
)



fixed_epsilon = 0.1
# ==============================
# ===== 2. WHITE-BOX АТАКИ =====
# ==============================
log_print("\n--- White-box атаки на нейронную сеть ---")
X_test_adv_fgsm = targeted_fgsm_attack(model, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                       feature_importance=importance_mask)
X_test_adv_pgd = targeted_pgd_attack(model, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon, alpha=0.01,
                                     num_iter=40, feature_importance=importance_mask)
X_test_adv_bim = targeted_bim_attack(model, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon, alpha=0.005,
                                     num_iter=10, feature_importance=importance_mask)

# Оценка атак на нейронную сеть
acc_nn_fgsm = evaluate_attack(model, X_test_adv_fgsm, X_test,y_test,y_test_tensor)
acc_nn_pgd = evaluate_attack(model, X_test_adv_pgd, X_test,y_test,y_test_tensor)
acc_nn_bim = evaluate_attack(model, X_test_adv_bim, X_test,y_test,y_test_tensor)

log_print(f"Точность нейронной сети при White-box FGSM атаке: {acc_nn_fgsm * 100:.2f}%")
log_print(f"Точность нейронной сети при White-box PGD атаке: {acc_nn_pgd * 100:.2f}%")
log_print(f"Точность нейронной сети при White-box BIM атаке: {acc_nn_bim * 100:.2f}%")

# Применение атак к модели бустинга
log_print("\n--- White-box атаки на модель бустинга ---")
# Преобразование PyTorch тензоров в numpy для бустинга
X_test_adv_fgsm_np = X_test_adv_fgsm.detach().cpu().numpy()
X_test_adv_pgd_np = X_test_adv_pgd.detach().cpu().numpy()
X_test_adv_bim_np = X_test_adv_bim.detach().cpu().numpy()

# Оценка атак на модель бустинга
acc_boost_fgsm = boost_model.score(X_test_adv_fgsm_np, y_test[:len(X_test_adv_fgsm_np)])
acc_boost_pgd = boost_model.score(X_test_adv_pgd_np, y_test[:len(X_test_adv_pgd_np)])
acc_boost_bim = boost_model.score(X_test_adv_bim_np, y_test[:len(X_test_adv_bim_np)])

log_print(f"Точность модели бустинга при White-box FGSM атаке: {acc_boost_fgsm * 100:.2f}%")
log_print(f"Точность модели бустинга при White-box PGD атаке: {acc_boost_pgd * 100:.2f}%")
log_print(f"Точность модели бустинга при White-box BIM атаке: {acc_boost_bim * 100:.2f}%")

# ==============================
# ===== 3. BLACK-BOX АТАКИ =====
# ==============================
log_print("\n--- Black-box атаки ---")
X_test_adv_transfer = targeted_transfer_attack(surrogate_model, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                               feature_importance=importance_mask)
X_test_adv_random = targeted_random_attack(model, X_test_tensor, y_test_tensor, epsilon=3, num_trials=200,
                                           feature_importance=importance_mask)
X_test_adv_square = targeted_square_attack(model, X_test_tensor, y_test_tensor, epsilon=3, iters=100,
                                           feature_importance=importance_mask)

# Оценка black-box атак на нейронную сеть
acc_nn_transfer = evaluate_attack(model, X_test_adv_transfer, X_test,y_test,y_test_tensor)
acc_nn_random = evaluate_attack(model, X_test_adv_random, X_test,y_test,y_test_tensor)
acc_nn_square = evaluate_attack(model, X_test_adv_square, X_test,y_test,y_test_tensor)

log_print(f"Точность нейронной сети при Black-box Transfer атаке: {acc_nn_transfer * 100:.2f}%")
log_print(f"Точность нейронной сети при Black-box Random атаке: {acc_nn_random * 100:.2f}%")
log_print(f"Точность нейронной сети при Black-box Square атаке: {acc_nn_square * 100:.2f}%")

# Оценка black-box атак на модель бустинга
X_test_adv_transfer_np = X_test_adv_transfer.detach().cpu().numpy()
X_test_adv_random_np = X_test_adv_random.detach().cpu().numpy()
X_test_adv_square_np = X_test_adv_square.detach().cpu().numpy()

acc_boost_transfer = boost_model.score(X_test_adv_transfer_np, y_test[:len(X_test_adv_transfer_np)])
acc_boost_random = boost_model.score(X_test_adv_random_np, y_test[:len(X_test_adv_random_np)])
acc_boost_square = boost_model.score(X_test_adv_square_np, y_test[:len(X_test_adv_square_np)])

log_print(f"Точность модели бустинга при Black-box Transfer атаке: {acc_boost_transfer * 100:.2f}%")
log_print(f"Точность модели бустинга при Black-box Random атаке: {acc_boost_random * 100:.2f}%")
log_print(f"Точность модели бустинга при Black-box Square атаке: {acc_boost_square * 100:.2f}%")

# ================================
# ===== 4. Защищённая модель =====
# ================================

model_def = MLP(input_dim, num_classes)
optimizer_def = optim.Adam(model_def.parameters(), lr=0.001)


# Создание защищенной модели бустинга
boost_model_def = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
purifier = FeaturePurifier(input_dim, latent_dim=32)
purifier_optimizer = optim.Adam(purifier.parameters(), lr=0.001)
purifier_criterion = nn.MSELoss()

model_purified = MLP(input_dim, num_classes)
optimizer_purified = optim.Adam(model_purified.parameters(), lr=0.001)
# Обучение очищителя
train_purifier(purifier, train_loader, purifier_optimizer, purifier_criterion, num_epochs_purifier, model=model, epsilon=0.05, importance_mask=importance_mask)

# Адверсариальное обучение
adversarial_training(model_def, train_loader, criterion, optimizer_def, num_epochs, attack_fn=mixed_adversarial_examples, purifier=purifier, epsilon=0.1)

adversarial_training(model_def, train_loader, criterion, optimizer_def, num_epochs_def, mixed_adversarial_examples,purifier=purifier)
# Обучение бустинг-модели на расширенных данных
train_boosting_with_adversarial_examples(boost_model_def, model, purifier, X_train, y_train, mixed_adversarial_examples, epsilon=0.1)
# Модель бустинга обученная на очищенных данных
boost_model_purified = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
# Создание очищенного набора данных
X_train_purified = create_purified_dataset(X_train_tensor, purifier)

boost_model_purified.fit(X_train_purified, y_train)
# Обучение модели на очищенных данных
train_model_with_purified_data(model_purified, X_train_purified, y_train, criterion, optimizer_purified, num_epochs)


# Оценка защищенных моделей без атак
acc_nn_def_before_attack = evaluate(model_def, X_test,y_test, test_loader)
acc_boost_def_before_attack = boost_model_def.score(X_test, y_test)
acc_boost_purified_before_attack = boost_model_purified.score(X_test, y_test)
acc_nn_purified_before_attack = evaluate(model_purified, X_test,y_test, test_loader)

log_print(f"Точность защищенной нейронной сети до атаки: {acc_nn_def_before_attack * 100:.2f}%")
log_print(f"Точность обычной модели на очищенных данных до атак: {acc_nn_purified_before_attack * 100:.2f}%")
log_print(f"Точность защищенной модели бустинга до атаки: {acc_boost_def_before_attack * 100:.2f}%")
log_print(f"Точность модели бустинга на очищенных данных до атаки: {acc_boost_purified_before_attack * 100:.2f}%")

log_print("\n--- Оценка защищённой модели под White-box атаками ---")
X_test_adv_fgsm_def = targeted_fgsm_attack(model_def, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                           feature_importance=importance_mask)
X_test_adv_pgd_def = targeted_pgd_attack(model_def, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon, alpha=0.01,
                                         num_iter=40, feature_importance=importance_mask)
X_test_adv_bim_def = targeted_bim_attack(model_def, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon, alpha=0.005,
                                         num_iter=10, feature_importance=importance_mask)

# Оценка атак на защищенную нейронную сеть
acc_fgsm_def_raw = evaluate_attack(model_def, X_test_adv_fgsm_def, X_test,y_test,y_test_tensor)
acc_pgd_def_raw = evaluate_attack(model_def, X_test_adv_pgd_def, X_test,y_test,y_test_tensor)
acc_bim_def_raw = evaluate_attack(model_def, X_test_adv_bim_def, X_test,y_test,y_test_tensor)

acc_fgsm_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_fgsm_def, y_test_tensor,y_test)
acc_pgd_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_pgd_def, y_test_tensor,y_test)
acc_bim_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_bim_def, y_test_tensor,y_test)

log_print(f"\nЗащищенная модель (исходная) - точность при White-box FGSM атаке: {acc_fgsm_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при White-box FGSM атаке: {acc_fgsm_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) - точность при White-box PGD атаке: {acc_pgd_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при White-box PGD атаке: {acc_pgd_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) - точность при White-box BIM атаке: {acc_bim_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при White-box BIM атаке: {acc_bim_def_purified * 100:.2f}%")

# Оценка атак на защищенную модель бустинга
X_test_adv_fgsm_np = X_test_adv_fgsm_def.detach().cpu().numpy()
X_test_adv_pgd_np = X_test_adv_pgd_def.detach().cpu().numpy()
X_test_adv_bim_np = X_test_adv_bim_def.detach().cpu().numpy()

acc_boost_fgsm_def = boost_model_def.score(X_test_adv_fgsm_np, y_test[:len(X_test_adv_fgsm_np)])
acc_boost_pgd_def = boost_model_def.score(X_test_adv_pgd_np, y_test[:len(X_test_adv_pgd_np)])
acc_boost_bim_def = boost_model_def.score(X_test_adv_bim_np, y_test[:len(X_test_adv_bim_np)])

acc_boost_fgsm_purified = boost_model_purified.score(X_test_adv_fgsm_np, y_test[:len(X_test_adv_fgsm_np)])
acc_boost_pgd_purified = boost_model_purified.score(X_test_adv_pgd_np, y_test[:len(X_test_adv_pgd_np)])
acc_boost_bim_purified = boost_model_purified.score(X_test_adv_bim_np, y_test[:len(X_test_adv_bim_np)])

log_print(f"\nЗащищенная модель бустинга (исходная) - точность при White-box FGSM атаке: {acc_boost_fgsm_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при White-box FGSM атаке: {acc_boost_fgsm_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) - точность при White-box PGD атаке: {acc_boost_pgd_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при White-box PGD атаке: {acc_boost_pgd_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) - точность при White-box BIM атаке: {acc_boost_bim_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при White-box BIM атаке: {acc_boost_bim_purified * 100:.2f}%")

# Оценка Black-box атак на защищенные модели
log_print("\n--- Оценка защищённой модели под Black-box атаками ---")

# Black-box Transfer атака
X_test_adv_transfer_def = targeted_transfer_attack(surrogate_model, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                                   feature_importance=importance_mask)

# Black-box Random атака
X_test_adv_random_def = targeted_random_attack(model_def, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                               num_trials=200, feature_importance=importance_mask)

# Black-box Square атака
X_test_adv_square_def = targeted_square_attack(model_def, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                               iters=100, feature_importance=importance_mask)

# Оценка Black-box атак на защищенную нейронную сеть
acc_transfer_def_raw = evaluate_attack(model_def, X_test_adv_transfer_def, X_test,y_test,y_test_tensor)
acc_random_def_raw = evaluate_attack(model_def, X_test_adv_random_def, X_test,y_test,y_test_tensor)
acc_square_def_raw = evaluate_attack(model_def, X_test_adv_square_def, X_test,y_test,y_test_tensor)

acc_transfer_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_transfer_def, y_test_tensor,y_test)
acc_random_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_random_def, y_test_tensor,y_test)
acc_square_def_purified = evaluate_with_purifier(model_def, purifier, X_test_adv_square_def, y_test_tensor,y_test)

log_print(f"\nЗащищенная модель (исходная) - точность при Black-box Transfer атаке: {acc_transfer_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при Black-box Transfer атаке: {acc_transfer_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) - точность при Black-box Random атаке: {acc_random_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при Black-box Random атаке: {acc_random_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) - точность при Black-box Square атаке: {acc_square_def_raw * 100:.2f}%")
log_print(f"Защищенная модель (очищенная) - точность при Black-box Square атаке: {acc_square_def_purified * 100:.2f}%")

# Оценка Black-box атак на защищенную модель бустинга
X_test_adv_transfer_np = X_test_adv_transfer_def.detach().cpu().numpy()
X_test_adv_random_np = X_test_adv_random_def.detach().cpu().numpy()
X_test_adv_square_np = X_test_adv_square_def.detach().cpu().numpy()

acc_boost_transfer_def = boost_model_def.score(X_test_adv_transfer_np, y_test[:len(X_test_adv_transfer_np)])
acc_boost_random_def = boost_model_def.score(X_test_adv_random_np, y_test[:len(X_test_adv_random_np)])
acc_boost_square_def = boost_model_def.score(X_test_adv_square_np, y_test[:len(X_test_adv_square_np)])

acc_boost_transfer_purified = boost_model_purified.score(X_test_adv_transfer_np, y_test[:len(X_test_adv_transfer_np)])
acc_boost_random_purified = boost_model_purified.score(X_test_adv_random_np, y_test[:len(X_test_adv_random_np)])
acc_boost_square_purified = boost_model_purified.score(X_test_adv_square_np, y_test[:len(X_test_adv_square_np)])

log_print(f"\nЗащищенная модель бустинга (исходная) - точность при Black-box Transfer атаке: {acc_boost_transfer_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при Black-box Transfer атаке: {acc_boost_transfer_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) - точность при Black-box Random атаке: {acc_boost_random_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при Black-box Random атаке: {acc_boost_random_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) - точность при Black-box Square атаке: {acc_boost_square_def * 100:.2f}%")
log_print(f"Защищенная модель бустинга (очищенная) - точность при Black-box Square атаке: {acc_boost_square_purified * 100:.2f}%")

# Оценка атак на модель бустинга
fgsm_results_boost = evaluate_boost_attack_with_epsilon(
    boost_model,
    targeted_fgsm_attack_boost,
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    y_test,
    importance_mask
)
pgd_results_boost = evaluate_boost_attack_with_epsilon(
    boost_model,
    lambda m, x, y, e, i: targeted_pgd_attack_boost(m, x, y, e, alpha=0.01, num_iter=10, feature_importance=i),
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    y_test,
    importance_mask
)
bim_results_boost = evaluate_boost_attack_with_epsilon(
    boost_model,
    lambda m, x, y, e, i: targeted_bim_attack_boost(m, x, y, e, alpha=0.005, num_iter=5, feature_importance=i),
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    y_test,
    importance_mask
)
random_results_boost = evaluate_boost_attack_with_epsilon(
    boost_model,
    lambda m, x, y, e, i: targeted_random_attack_boost(m, x, y, e, num_trials=50, feature_importance=i),
    epsilon_values,
    X_test_tensor,
    y_test_tensor,
    y_test,
    importance_mask
)

# White-box FGSM атака на очищенные данные
X_test_adv_fgsm_purified = targeted_fgsm_attack(model_purified, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                                feature_importance=importance_mask)

# White-box PGD атака на очищенные данные
X_test_adv_pgd_purified = targeted_pgd_attack(model_purified, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                              alpha=0.01, num_iter=40, feature_importance=importance_mask)

# White-box BIM атака на очищенные данные
X_test_adv_bim_purified = targeted_bim_attack(model_purified, X_test_tensor, y_test_tensor, epsilon=fixed_epsilon,
                                              alpha=0.005, num_iter=10, feature_importance=importance_mask)


# =======================================================
# ===== 5. Анализ результатов и сохранение графиков =====
# =======================================================

plot_epsilon_results(fgsm_results, "FGSM", "Neural Network")
plot_epsilon_results(pgd_results, "PGD", "Neural Network")
plot_epsilon_results(bim_results, "BIM", "Neural Network")
plot_epsilon_results(random_results, "Random Search", "Neural Network")

plot_epsilon_results(fgsm_results_boost, "FGSM", "Boosting Model")
plot_epsilon_results(pgd_results_boost, "PGD", "Boosting Model")
plot_epsilon_results(bim_results_boost, "BIM", "Boosting Model")
plot_epsilon_results(random_results_boost, "Random Search", "Boosting Model")

plot_confusion_matrix(model_def, X_test_tensor, y_test_tensor, "Defended Neural Network - Before Attacks")
plot_confusion_matrix(model_def, X_test_adv_fgsm_def, y_test_tensor,
                      "Defended Neural Network - White-box FGSM Attack (Raw)")
plot_confusion_matrix(model_def, purifier(X_test_adv_fgsm_def), y_test_tensor,
                      "Defended Neural Network - White-box FGSM Attack (Purified)")

plot_confusion_matrix(model_purified, X_test_tensor, y_test_tensor, "Purified Neural Network - Before Attacks")
plot_confusion_matrix(model_purified, X_test_adv_fgsm_purified, y_test_tensor, "Purified Neural Network - White-box FGSM Attack (Raw)")
plot_confusion_matrix(model_purified, purifier(X_test_adv_fgsm_purified), y_test_tensor,
                      "Purified Neural Network - White-box FGSM Attack (Purified)")

plot_confusion_matrix(boost_model_def, X_test_tensor, y_test_tensor, "Defended Boosting Model - Before Attacks")
plot_confusion_matrix(boost_model_def, X_test_adv_fgsm_def, y_test_tensor,
                      "Defended Boosting Model - White-box FGSM Attack (Raw)")
plot_confusion_matrix(boost_model_def, purifier(X_test_adv_fgsm_def), y_test_tensor,
                      "Defended Boosting Model - White-box FGSM Attack (Purified)")

plot_confusion_matrix(boost_model_purified, X_test_tensor, y_test_tensor, "Purified Boosting Model - Before Attacks")
plot_confusion_matrix(boost_model_purified, X_test_adv_fgsm_def, y_test_tensor,
                      "Purified Boosting Model - White-box FGSM Attack (Raw)")
plot_confusion_matrix(boost_model_purified, purifier(X_test_adv_fgsm_def), y_test_tensor,
                      "Purified Boosting Model - White-box FGSM Attack (Purified)")
# Итоговое сравнение результатов атак
log_print("\n--- Итоговое сравнение результатов атак ---")
log_print(f"Защищенная модель (исходная) FGSM: {acc_fgsm_def_raw * 100:.2f}% | Очищенная: {acc_fgsm_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) PGD:  {acc_pgd_def_raw * 100:.2f}% | Очищенная: {acc_pgd_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) BIM:  {acc_bim_def_raw * 100:.2f}% | Очищенная: {acc_bim_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) Transfer: {acc_transfer_def_raw * 100:.2f}% | Очищенная: {acc_transfer_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) Random:   {acc_random_def_raw * 100:.2f}% | Очищенная: {acc_random_def_purified * 100:.2f}%")
log_print(f"Защищенная модель (исходная) Square:   {acc_square_def_raw * 100:.2f}% | Очищенная: {acc_square_def_purified * 100:.2f}%")

log_print(f"\nЗащищенная модель бустинга (исходная) FGSM: {acc_boost_fgsm_def * 100:.2f}% | Очищенная: {acc_boost_fgsm_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) PGD:  {acc_boost_pgd_def * 100:.2f}% | Очищенная: {acc_boost_pgd_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) BIM:  {acc_boost_bim_def * 100:.2f}% | Очищенная: {acc_boost_bim_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) Transfer: {acc_boost_transfer_def * 100:.2f}% | Очищенная: {acc_boost_transfer_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) Random:   {acc_boost_random_def * 100:.2f}% | Очищенная: {acc_boost_random_purified * 100:.2f}%")
log_print(f"Защищенная модель бустинга (исходная) Square:   {acc_boost_square_def * 100:.2f}% | Очищенная: {acc_boost_square_purified * 100:.2f}%")
