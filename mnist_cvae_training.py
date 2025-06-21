
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 50
latent_dim = 20
num_classes = 10

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 7x7 -> 3x3
            nn.ReLU(),
        )

        # Calculate flattened size after conv layers
        self.conv_output_size = 128 * 3 * 3

        # Encoder fully connected layers
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.conv_output_size + num_classes, 256),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, self.conv_output_size),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 3x3 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        x = self.encoder_fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, 3, 3)
        x = self.decoder_conv(x)
        return x

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train_model():
    model = ConditionalVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            labels = labels.to(device)

            # One-hot encode labels
            labels_onehot = F.one_hot(labels, num_classes=num_classes).float()

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data, labels_onehot)
            loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item() / len(data)})

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'cvae_epoch_{epoch+1}.pth')

    # Save final model
    torch.save(model.state_dict(), 'cvae_final.pth')
    return model, train_losses

def generate_digits(model, digit, num_samples=5):
    """Generate specified number of digit samples"""
    model.eval()
    with torch.no_grad():
        # Create one-hot encoded labels
        labels = torch.tensor([digit] * num_samples).to(device)
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float()

        # Sample from latent space
        z = torch.randn(num_samples, latent_dim).to(device)

        # Generate images
        generated = model.decode(z, labels_onehot)

    return generated.cpu()

def visualize_generation(model, digit):
    """Visualize generated digits"""
    generated = generate_digits(model, digit, 10)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].squeeze(), cmap='gray')
        ax.set_title(f'Generated {digit}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'generated_digit_{digit}.png')
    plt.show()

def main():
    print("Starting CVAE training for MNIST digit generation...")

    # Train the model
    model, losses = train_model()

    # Generate samples for each digit
    print("\nGenerating sample digits...")
    for digit in range(10):
        visualize_generation(model, digit)

    print("Training completed! Model saved as 'cvae_final.pth'")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()
