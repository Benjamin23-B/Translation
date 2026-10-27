"""
Batch translation verification script.
Tests individual words, phrases, and complete sentences.
"""

import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from translate import Translator

def main():
    translator = Translator()

    test_cases = [
        # Words
        "hello",
        "welcome",
        "thank you",
        "water",
        "food",
        "tea",
        "father",
        "mother",
        "friend",
        "today",
        "tomorrow",
        "happy",
        
        # Phrases & Sentences
        "good morning",
        "good evening",
        "how are you",
        "what is your name",
        "i love tamil",
        "i speak english",
        "i am learning tamil",
        "where is the school",
        "where are you",
        "thank you very much",
        "i am hungry",
        "i want water",
        "open the door",
        "this book is good",
        "it is raining",
        "today is a good day"
    ]

    print("\n" + "=" * 70)
    print(f"{'ENGLISH INPUT':<32} | {'TAMIL TRANSLATION'}")
    print("=" * 70)

    for en_text in test_cases:
        tamil_out = translator.translate(en_text)
        print(f"{en_text:<32} | {tamil_out}")

    print("=" * 70)

if __name__ == "__main__":
    main()
