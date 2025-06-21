# Handwritten Digit Generator Web App

A web application that generates handwritten digits using a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset.

## Quick Start

### 1. Train the Model (Google Colab)
```bash
# Upload mnist_cvae_training.py to Google Colab
# Run the training script
python mnist_cvae_training.py
```

### 2. Deploy the Web App

#### Option A: Streamlit Cloud (Recommended)
1. Upload files to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy automatically

#### Option B: Hugging Face Spaces
1. Create new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload all files
3. Select Streamlit as framework

#### Option C: Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files Description

- `mnist_cvae_training.py` - Training script for CVAE model
- `streamlit_app.py` - Web application code
- `requirements.txt` - Python dependencies
- `cvae_final.pth` - Trained model weights (generated after training)

## Model Architecture

**Conditional Variational Autoencoder (CVAE)**
- Latent dimension: 20
- Input/Output: 28x28 grayscale images
- Conditional on digit labels (0-9)
- Training: 50 epochs on MNIST dataset

## Training Requirements

- Google Colab with T4 GPU (free tier sufficient)
- Training time: ~30-45 minutes
- Dataset: MNIST (automatically downloaded)

## Web App Features

- Select digit 0-9 from dropdown
- Generate 5 unique samples per request
- Display both grid view and individual images
- Responsive design with technical details

## Deployment Notes

- Ensure `cvae_final.pth` is uploaded with the web app
- App will show error if model file is missing
- Public access required for 2+ weeks
- Cold start acceptable (app can sleep when idle)

## Example Usage

1. Select digit from dropdown (e.g., "7")
2. Click "Generate 5 Samples"
3. View generated handwritten digit images
4. Each generation produces unique variations

## Technical Details

- Framework: PyTorch + Streamlit
- Model size: ~2MB
- Generation time: <1 second
- Image format: 28x28 grayscale PNG
- Compatible with MNIST format
