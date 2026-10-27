"""
Training script for the English-to-Tamil Transformer translation model.
Uses custom learning rate schedule, masked loss/accuracy, and tf.GradientTape training step.
"""

import os
import time
import numpy as np
import tensorflow as tf

from dataset import load_dataset
from tokenizer import Tokenizer, PAD_ID, START_ID, END_ID
from transformer import Transformer, create_padding_mask, create_look_ahead_mask

# ----------------- Hyperparameters -----------------
MAX_LEN = 35
NUM_LAYERS = 2
D_MODEL = 128
DFF = 512
NUM_HEADS = 4
DROPOUT_RATE = 0.1
BATCH_SIZE = 16
EPOCHS = 100
WARMUP_STEPS = 150
SAVE_DIR = "saved_model"


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=400):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": int(self.d_model), "warmup_steps": self.warmup_steps}


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, PAD_ID))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype))
    mask = tf.math.logical_not(tf.math.equal(real, PAD_ID))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def prepare_dataset(pairs, token_en, token_ta, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    en_texts = [p[0] for p in pairs]
    ta_texts = [p[1] for p in pairs]

    en_encoded = np.array(
        [token_en.encode(t, add_special=True, max_len=max_len) for t in en_texts],
        dtype=np.int32,
    )
    ta_encoded = np.array(
        [token_ta.encode(t, add_special=True, max_len=max_len) for t in ta_texts],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices((en_encoded, ta_encoded))
    dataset = dataset.shuffle(buffer_size=len(pairs))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load and prepare dataset
    print("=" * 60)
    print("Loading English-Tamil dataset...")
    pairs = load_dataset()
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    split = int(0.9 * len(pairs))
    train_pairs = [pairs[i] for i in indices[:split]]
    val_pairs = [pairs[i] for i in indices[split:]]

    print(f"Training pairs: {len(train_pairs)} | Validation pairs: {len(val_pairs)}")

    # 2. Tokenizers
    print("Fitting tokenizers...")
    token_en = Tokenizer(lang="en")
    token_en.fit_on_texts([p[0] for p in pairs])

    token_ta = Tokenizer(lang="ta")
    token_ta.fit_on_texts([p[1] for p in pairs])

    token_en.save(os.path.join(SAVE_DIR, "tokenizer_en.json"))
    token_ta.save(os.path.join(SAVE_DIR, "tokenizer_ta.json"))

    print(f"English Vocab Size: {token_en.vocab_size}")
    print(f"Tamil Vocab Size:   {token_ta.vocab_size}")

    train_dataset = prepare_dataset(train_pairs, token_en, token_ta)
    val_dataset = prepare_dataset(val_pairs, token_en, token_ta)

    # 3. Instantiate Transformer
    print("=" * 60)
    print("Building lightweight Transformer Model...")
    learning_rate = CustomSchedule(D_MODEL, warmup_steps=WARMUP_STEPS)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=token_en.vocab_size + 10,
        target_vocab_size=token_ta.vocab_size + 10,
        pe_input=MAX_LEN + 10,
        pe_target=MAX_LEN + 10,
        rate=DROPOUT_RATE,
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions = transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    @tf.function
    def eval_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        predictions = transformer([inp, tar_inp], training=False)
        loss = loss_function(tar_real, predictions)

        val_loss(loss)
        val_accuracy(accuracy_function(tar_real, predictions))

    # Initialize model weights with dummy input
    dummy_inp = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    dummy_tar = tf.zeros((1, MAX_LEN - 1), dtype=tf.int32)
    _ = transformer([dummy_inp, dummy_tar], training=False)
    
    total_params = sum(int(np.prod(v.shape)) for v in transformer.trainable_variables)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 60)

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        epoch_start = time.time()

        for batch, (inp, tar) in enumerate(train_dataset):
            train_step(inp, tar)

        for inp, tar in val_dataset:
            eval_step(inp, tar)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
                f"Train Loss: {train_loss.result():.4f} | "
                f"Train Acc: {train_accuracy.result()*100:.2f}% | "
                f"Val Loss: {val_loss.result():.4f} | "
                f"Val Acc: {val_accuracy.result()*100:.2f}% | "
                f"Time: {time.time() - epoch_start:.2f}s"
            )

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 5. Save model weights
    weights_path = os.path.join(SAVE_DIR, "transformer.weights.h5")
    transformer.save_weights(weights_path)
    print(f"Model weights successfully saved to {weights_path}")
    print("=" * 60)
    return transformer, token_en, token_ta


if __name__ == "__main__":
    train()
