"""
Tokenizer module for English and Tamil.
Built using standard Python, NumPy, and JSON serialization.
Handles tokenization, vocabulary creation, sequence padding, encoding, and decoding.
"""

import re
import json
import numpy as np

# Special token constants
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


def clean_text(text):
    """Normalizes whitespace and removes unwanted control characters."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_english(text):
    """Tokenizes English sentence into words and punctuation."""
    text = text.lower().strip()
    # Separate punctuation
    text = re.sub(r"([?.!,;:\'\"()\[\]])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return [t for t in text.strip().split(" ") if t]


def tokenize_tamil(text):
    """Tokenizes Tamil sentence into words and punctuation while preserving Tamil Unicode."""
    text = text.strip()
    # Separate punctuation while keeping Tamil characters intact
    text = re.sub(r"([?.!,;:\'\"()\[\]])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return [t for t in text.strip().split(" ") if t]


class Tokenizer:
    def __init__(self, lang="en", min_freq=1):
        self.lang = lang
        self.min_freq = min_freq
        self.word2idx = {
            PAD_TOKEN: PAD_ID,
            START_TOKEN: START_ID,
            END_TOKEN: END_ID,
            UNK_TOKEN: UNK_ID,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text):
        if self.lang == "en":
            return tokenize_english(text)
        else:
            return tokenize_tamil(text)

    def fit_on_texts(self, texts):
        word_counts = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        for token, count in word_counts.items():
            if count >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

        self.vocab_size = len(self.word2idx)
        return self

    def encode(self, text, add_special=True, max_len=None):
        tokens = self.tokenize(text)
        token_ids = []
        if add_special:
            token_ids.append(START_ID)

        for token in tokens:
            token_ids.append(self.word2idx.get(token, UNK_ID))

        if add_special:
            token_ids.append(END_ID)

        if max_len is not None:
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                if add_special and token_ids[-1] != END_ID:
                    token_ids[-1] = END_ID
            else:
                token_ids = token_ids + [PAD_ID] * (max_len - len(token_ids))

        return token_ids

    def decode(self, token_ids, remove_special=True):
        words = []
        for idx in token_ids:
            if isinstance(idx, (np.integer, int)):
                idx_val = int(idx)
            else:
                idx_val = int(idx.numpy()) if hasattr(idx, 'numpy') else int(idx)

            if idx_val == PAD_ID and remove_special:
                continue
            if idx_val == START_ID and remove_special:
                continue
            if idx_val == END_ID and remove_special:
                break

            word = self.idx2word.get(idx_val, UNK_TOKEN)
            words.append(word)

        # Re-join tokens smoothly
        result = " ".join(words)
        # Fix punctuation spacing
        result = re.sub(r"\s+([?.!,;:\'\"])", r"\1", result)
        return result.strip()

    def save(self, file_path):
        data = {
            "lang": self.lang,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(lang=data["lang"], min_freq=data["min_freq"])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tokenizer.vocab_size = len(tokenizer.word2idx)
        return tokenizer
