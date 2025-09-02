# Vision Transformer for MNIST Classification

This repository contains a PyTorch implementation of a Vision Transformer (ViT) model for image classification on the MNIST dataset.

## Project Structure

vision-transformer-/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── src/
│   ├── init.py
│   ├── model.py
│   ├── dataset.py
│   ├── training.py
│   └── visualize.py
└── notebooks/
   └── vision transformer.ipynb


## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Abel-Marie/vision_transformer-.git](.git)
    cd vision-transformer-
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the model, run the main script:

```bash
python main.py