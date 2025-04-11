# GPT-Transformer under the guide of paper attention is all you need(2017)

A simple implementation of a character-level transformer language model using PyTorch. This project demonstrates the fundamentals of training a Generative Pre-trained Transformer (GPT) model from scratch on a custom text corpus.

## 🔧 Features

- Character-level tokenization
- Multi-head self-attention mechanism
- Positional encoding
- Stack of transformer blocks
- Training and validation loss estimation
- Text generation capability
```
## 🗂️ Project Structure
📁 gpt-transformer/ 
  ├── gpt.py # Main implementation of the GPT model, train, generation
  ├── input.txt # Training input data
  ├── output.txt # Generated output from the model
  ├── gpt-transformer.pth # Example of trained model weights
  └── README.md # Project documentation
```
## 🚀 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/tarek1488/gpt-transformer.git
cd gpt-transformer
python gpt.py


