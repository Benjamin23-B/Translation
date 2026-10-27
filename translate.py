"""
Inference & Translation Module for English-to-Tamil Translation.
Loads the trained Transformer weights and tokenizers to perform autoregressive decoding.
Supports single words, phrases, full sentences, batch translation, and interactive CLI.
"""

import os
import sys
import io

# Ensure UTF-8 output on Windows console for Tamil characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import numpy as np
import tensorflow as tf

from tokenizer import Tokenizer, PAD_ID, START_ID, END_ID
from transformer import Transformer

MAX_LEN = 35
NUM_LAYERS = 2
D_MODEL = 128
DFF = 512
NUM_HEADS = 4
SAVE_DIR = "saved_model"


class Translator:
    def __init__(self, model_dir=SAVE_DIR):
        self.model_dir = model_dir
        self.token_en_path = os.path.join(model_dir, "tokenizer_en.json")
        self.token_ta_path = os.path.join(model_dir, "tokenizer_ta.json")
        self.weights_path = os.path.join(model_dir, "transformer.weights.h5")

        if not os.path.exists(self.token_en_path) or not os.path.exists(self.token_ta_path):
            raise FileNotFoundError(
                f"Tokenizer files not found in '{model_dir}'. Please run train.py first!"
            )

        print("Loading tokenizers...")
        self.token_en = Tokenizer.load(self.token_en_path)
        self.token_ta = Tokenizer.load(self.token_ta_path)

        print("Building Transformer model...")
        self.transformer = Transformer(
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            input_vocab_size=self.token_en.vocab_size + 10,
            target_vocab_size=self.token_ta.vocab_size + 10,
            pe_input=MAX_LEN + 10,
            pe_target=MAX_LEN + 10,
        )

        # Build model shapes
        dummy_inp = tf.zeros((1, MAX_LEN), dtype=tf.int32)
        dummy_tar = tf.zeros((1, MAX_LEN - 1), dtype=tf.int32)
        _ = self.transformer([dummy_inp, dummy_tar], training=False)

        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}...")
            self.transformer.load_weights(self.weights_path)
            print("Model successfully loaded!")
        else:
            raise FileNotFoundError(
                f"Model weights not found at '{self.weights_path}'. Please run train.py first!"
            )

    def translate(self, sentence, max_length=MAX_LEN):
        """
        Translates an English sentence or word to Tamil using autoregressive greedy decoding.
        """
        sentence = sentence.strip()
        if not sentence:
            return ""

        # 1. Encode English sentence
        encoder_input = self.token_en.encode(
            sentence, add_special=True, max_len=max_length
        )
        encoder_input = tf.constant([encoder_input], dtype=tf.int32)

        # 2. Initialize output with START token
        output_tokens = [START_ID]
        output = tf.constant([output_tokens], dtype=tf.int32)

        for _ in range(max_length):
            # Shape: (1, current_seq_len, target_vocab_size)
            predictions = self.transformer([encoder_input, output], training=False)

            # Select last token from seq_len dimension
            predicted_id = tf.cast(
                tf.argmax(predictions[:, -1, :], axis=-1), dtype=tf.int32
            )
            predicted_id = int(predicted_id.numpy()[0])

            # Stop if END token is reached
            if predicted_id == END_ID:
                break

            output_tokens.append(predicted_id)
            output = tf.constant([output_tokens], dtype=tf.int32)

        # Decode tokens back to Tamil string
        translated_text = self.token_ta.decode(output_tokens, remove_special=True)
        return translated_text


def run_interactive():
    print("=" * 60)
    print("English to Tamil Transformer Translation CLI")
    print("Type your English words or sentences to translate to Tamil.")
    print("Type 'q' or 'exit' to quit.")
    print("=" * 60)

    translator = Translator()

    # Pre-test samples
    test_samples = [
        "hello",
        "good morning",
        "how are you",
        "what is your name",
        "i love tamil",
        "where is the school",
        "thank you very much",
        "i want water",
        "today is a good day",
    ]

    print("\n--- Running Sample Translations ---")
    for sample in test_samples:
        result = translator.translate(sample)
        print(f"EN: {sample:25s} -> TA: {result}")
    print("-----------------------------------\n")

    while True:
        try:
            user_input = input("Enter English text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "exit", "quit"):
                print("Exiting translator. Goodbye!")
                break

            tamil_translation = translator.translate(user_input)
            print(f"Tamil: {tamil_translation}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting translator.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_translate = " ".join(sys.argv[1:])
        translator = Translator()
        translated = translator.translate(text_to_translate)
        print(f"\nInput English:  {text_to_translate}")
        print(f"Output Tamil:   {translated}\n")
    else:
        run_interactive()
