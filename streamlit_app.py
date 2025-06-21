
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.conv_output_size = 128 * 3 * 3

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.conv_output_size + num_classes, 256),
            nn.ReLU(),
        )

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

@st.cache_resource
def load_model():
    """Load the trained CVAE model"""
    model = ConditionalVAE(latent_dim=20, num_classes=10)
    try:
        # Try to load the model weights
        model.load_state_dict(torch.load('cvae_final.pth', map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'cvae_final.pth' not found. Please upload the trained model.")
        return None

def generate_digits(model, digit, num_samples=5):
    """Generate specified number of digit samples"""
    if model is None:
        return None

    with torch.no_grad():
        # Create one-hot encoded labels
        labels = torch.tensor([digit] * num_samples).to(device)
        labels_onehot = F.one_hot(labels, num_classes=10).float()

        # Sample from latent space with some randomness for diversity
        z = torch.randn(num_samples, 20).to(device)

        # Generate images
        generated = model.decode(z, labels_onehot)

    return generated.cpu().numpy()

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()

    return buf

def main():
    st.title("🔢 Handwritten Digit Generator")
    st.markdown("Generate handwritten digits using a Conditional Variational Autoencoder (CVAE) trained on MNIST dataset")

    # Sidebar
    st.sidebar.header("Settings")
    st.sidebar.markdown("Select a digit to generate 5 unique samples")

    # Model loading
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.error("❌ Could not load the trained model. Please ensure 'cvae_final.pth' is available.")
        st.info("To use this app, you need to first train the model using the provided training script.")
        return

    st.success("✅ Model loaded successfully!")

    # Digit selection
    col1, col2 = st.columns([1, 3])

    with col1:
        selected_digit = st.selectbox(
            "Choose a digit to generate:",
            options=list(range(10)),
            index=0,
            help="Select any digit from 0 to 9"
        )

        generate_button = st.button(
            "🎲 Generate 5 Samples",
            type="primary",
            help="Click to generate 5 unique samples of the selected digit"
        )

    with col2:
        if generate_button:
            with st.spinner(f"Generating digit {selected_digit}..."):
                # Generate images
                generated_images = generate_digits(model, selected_digit, 5)

                if generated_images is not None:
                    # Create and display image grid
                    img_buffer = create_image_grid(generated_images)
                    st.image(img_buffer, caption=f"Generated samples of digit {selected_digit}")

                    # Show individual images in columns
                    st.subheader("Individual Samples:")
                    cols = st.columns(5)

                    for i in range(5):
                        with cols[i]:
                            # Convert numpy array to PIL Image for better display
                            img_array = (generated_images[i].squeeze() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_array, mode='L')
                            st.image(pil_img, caption=f"Sample {i+1}", width=100)
                else:
                    st.error("Failed to generate images")

    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This App")
    st.markdown("""
    This app uses a **Conditional Variational Autoencoder (CVAE)** trained from scratch on the MNIST dataset to generate handwritten digits.

    **Key Features:**
    - Generates 5 unique samples for any selected digit (0-9)
    - Uses a deep learning model trained specifically for this task
    - 28x28 grayscale images similar to the original MNIST format
    - Each generation produces diverse samples with natural handwriting variations

    **Model Architecture:**
    - Encoder: Convolutional layers + fully connected layers
    - Latent space: 20-dimensional with conditional information
    - Decoder: Fully connected layers + transposed convolutional layers
    - Training: 50 epochs on full MNIST training set
    """)

    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Specifications:**
        - Framework: PyTorch
        - Architecture: Conditional VAE
        - Latent Dimension: 20
        - Input/Output: 28x28 grayscale images
        - Loss Function: Reconstruction loss (BCE) + KL divergence
        - Training Device: Google Colab T4 GPU

        **Generation Process:**
        1. Sample random noise from latent space
        2. Combine with one-hot encoded digit label
        3. Pass through decoder network
        4. Output 28x28 probability map
        5. Convert to grayscale image
        """)

if __name__ == "__main__":
    main()
