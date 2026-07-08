# 🧠 Vision Transformer for MNIST Classification

A PyTorch implementation of a **Vision Transformer (ViT)** for image classification on the **MNIST** dataset — demonstrating how transformer architectures, originally built for NLP, apply to computer vision.

## 📘 About

Vision Transformers process an image by **splitting it into fixed-size patches**, linearly projecting each patch into an embedding, and feeding the sequence into a standard transformer encoder. Unlike CNNs, ViTs capture **global context** across the whole image from the first layer via self-attention.

Key components implemented here:
- **Patch embedding** — the image is split into fixed-size patches, each projected into an embedding vector.
- **Positional encoding** — added so the model retains the spatial position of each patch.
- **Transformer encoder** — multi-head self-attention + feed-forward layers over the patch sequence.
- **[CLS] token** — a classification token prepended to the sequence, used for the final prediction.
- **Classification head** — maps the [CLS] embedding to class logits.

## 📁 Project structure

```text
vision_transformer-/
├── main.py                # Entry point: train + evaluate
├── config.py              # Hyperparameters / config
├── requirements.txt
└── src/
    ├── model.py           # Vision Transformer architecture
    ├── dataset.py         # MNIST loading + transforms
    ├── training.py        # Training / evaluation loop
    └── visualize.py       # (Optional) attention visualization
```

## ⚙️ Setup

```bash
git clone https://github.com/Abel-Marie/vision_transformer-.git
cd vision_transformer-
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python main.py
```

This loads and preprocesses MNIST, trains the ViT, and evaluates test accuracy (with optional attention-map visualization if enabled in the config).

## 📊 Results

The model reaches competitive accuracy on MNIST, showing that a transformer — with no convolutions — can classify images effectively even in a low-data regime.

## 📄 License

MIT
