# English to Tamil Neural Machine Translation with Transformer

A lightweight, high-performance English-to-Tamil machine translation system built entirely using **TensorFlow** and **NumPy** from scratch without high-level NLP libraries or UI dependencies.

---

## 🌟 Architecture Overview

- **Model Type**: Sequence-to-Sequence Transformer (Encoder-Decoder)
- **Frameworks**: TensorFlow 2.x & NumPy only
- **Lightweight Parameter Budget**: ~1,001,673 parameters (~1.0M params)
- **Model Hyperparameters**:
  - Embedding / Hidden Dimension ($d_{model}$): `128`
  - Feed-Forward Dimension ($d_{ff}$): `512`
  - Attention Heads ($h$): `4`
  - Encoder Layers: `2`
  - Decoder Layers: `2`
  - Max Sequence Length: `35` tokens
  - Dropout: `0.1`

---

## 📁 Project Structure

```
├── dataset.py            # English-Tamil parallel pairs & custom dataset loader
├── tokenizer.py          # Custom Tokenizer with special tokens (<pad>, <start>, <end>, <unk>)
├── transformer.py        # Positional Encoding, MultiHeadAttention, Encoder, Decoder, Transformer
├── train.py              # Custom training loop with masked loss/accuracy & lr scheduler
├── translate.py          # Autoregressive greedy inference & interactive CLI
├── test_translation.py   # Batch test verification suite
└── saved_model/          # Saved model weights & tokenizer vocabularies
    ├── tokenizer_en.json
    ├── tokenizer_ta.json
    └── transformer.weights.h5
```

---

## 🚀 How to Use

### 1. Translate a Single Sentence or Word
```bash
python translate.py "good morning"
python translate.py "what is your name"
python translate.py "water"
```

### 2. Interactive CLI Translation Mode
Launch the interactive shell to translate in real-time:
```bash
python translate.py
```
Type your English sentence or word, and press Enter to receive the Tamil translation. Type `q` or `exit` to quit.

### 3. Run Batch Translation Tests
```bash
python test_translation.py
```

### 4. Retrain the Model
```bash
python train.py
```

---

## 🛠 Adding Custom / External Datasets
You can supply your own TSV or CSV English-Tamil sentence pair file to `load_dataset("path/to/dataset.tsv")` in [dataset.py](file:///c:/Users/murug/Downloads/Translation/dataset.py) with the format:
```tsv
english_sentence	tamil_sentence
```
