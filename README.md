
---

```markdown
# 🧠 Vision Transformer for MNIST Classification

This repository contains a PyTorch implementation of a **Vision Transformer (ViT)** model for image classification on the **MNIST** dataset. It demonstrates how transformer-based architectures, originally developed for NLP, can be applied to computer vision tasks with strong performance and scalability.

---

## 📘 About Vision Transformers

Vision Transformers (ViT) introduce a new way of processing image data by **dividing images into fixed-size patches**, flattening them, and feeding them into a standard transformer encoder (like those used in language models such as BERT). Unlike CNNs, which rely on local receptive fields and convolutions, ViTs **capture global context** across the entire image from the very beginning using **self-attention**.

Key concepts:
- **Patch Embedding:** Images are split into fixed-size patches (e.g., 16x16), each of which is linearly projected into an embedding vector.
- **Positional Encoding:** Since transformers lack spatial awareness, positional encodings are added to maintain the relative position of patches.
- **Transformer Encoder:** Each patch embedding (with position info) is processed through multi-head self-attention and feed-forward layers.
- **[CLS] Token:** A special token is prepended to the patch sequence and used for classification, similar to NLP models.
- **Output Head:** After processing, the final [CLS] token embedding is passed through a classification head to predict the output class.

ViT models can outperform traditional convolutional neural networks, especially when trained on large datasets, and they provide better scalability and modeling of long-range dependencies.

---

## 📁 Project Structure

```

vision-transformer-/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── src/
│   ├── **init**.py
│   ├── model.py          # Vision Transformer architecture
│   ├── dataset.py        # MNIST data loader and transformations
│   ├── training.py       # Training and evaluation logic
│   └── visualize.py      # Optional visualization tools
└── notebooks/
└── vision\_transformer.ipynb  # Jupyter notebook version

````

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Abel-Marie/vision_transformer-.git
cd vision-transformer-
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To train and evaluate the Vision Transformer model on MNIST:

```bash
python main.py
```

This will:

* Load and preprocess the MNIST dataset
* Train the ViT model for digit classification
* Evaluate accuracy on the test set
* (Optional) Visualize attention maps if enabled in the config

---

## 📊 Results

The Vision Transformer model achieves competitive accuracy on MNIST and demonstrates the ability of transformers to handle image-based tasks efficiently, even in low-data regimes like MNIST.

---

## 🤝 Contributing

Pull requests and improvements are welcome! Please feel free to fork the repository, make changes, and submit a PR.

---

## 📄 License


```


