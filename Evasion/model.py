import torch
import torch.nn as nn

# Модель многослойного перцептрона (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        # Слои сети
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Прямой проход
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Модель суррогата для MLP
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SurrogateMLP, self).__init__()
        # Слои сети
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Прямой проход
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Класс для очистки признаков
class FeaturePurifier(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(FeaturePurifier, self).__init__()
        # Энкодер (сжатие признаков)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
        # Декодер (восстановление признаков)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim),
        )

    def forward(self, x):
        # Прямой проход: энкодер и декодер
        latent = self.encoder(x)
        purified = self.decoder(latent)
        return purified
