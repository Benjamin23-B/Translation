"""
English to Tamil Dataset Module
Provides rich parallel sentence pairs covering words, phrases, greetings, questions,
everyday conversation, numbers, and grammar patterns.
Also supports loading external dataset files (TSV, CSV, or raw text).
"""

import os
import re
import numpy as np

# Rich built-in English to Tamil parallel corpus
DEFAULT_PARALLEL_DATA = [
    # Basic Words and Greetings
    ("hello", "வணக்கம்"),
    ("hi", "வணக்கம்"),
    ("welcome", "நல்வரவு"),
    ("good morning", "காலை வணக்கம்"),
    ("good afternoon", "மதிய வணக்கம்"),
    ("good evening", "மாலை வணக்கம்"),
    ("good night", "இனிய இரவு"),
    ("thank you", "நன்றி"),
    ("thanks", "நன்றி"),
    ("thank you very much", "மிக்க நன்றி"),
    ("sorry", "மன்னிக்கவும்"),
    ("please", "தயவுசெய்து"),
    ("yes", "ஆம்"),
    ("no", "இல்லை"),
    ("okay", "சரி"),
    ("goodbye", "சென்று வருகிறேன்"),
    ("bye", "சென்று வருகிறேன்"),
    ("see you again", "மீண்டும் சந்திப்போம்"),

    # Pronouns & Questions
    ("who", "யார்"),
    ("what", "என்ன"),
    ("when", "எப்போது"),
    ("where", "எங்கே"),
    ("why", "ஏன்"),
    ("how", "எப்படி"),
    ("which", "எது"),
    ("how much", "எவ்வளவு"),
    ("how many", "எத்தனை"),
    ("who is this", "இவர் யார்"),
    ("what is this", "இது என்ன"),
    ("what is that", "அது என்ன"),
    ("where are you", "நீங்கள் எங்கே இருக்கிறீர்கள்"),
    ("where is the house", "வீடு எங்கே இருக்கிறது"),
    ("where is the school", "பள்ளி எங்கே இருக்கிறது"),
    ("where is the hospital", "மருத்துவமனை எங்கே இருக்கிறது"),
    ("why are you late", "நீங்கள் ஏன் தாமதமாக வந்தீர்கள்"),
    ("what is your name", "உங்கள் பெயர் என்ன"),
    ("how are you", "நீங்கள் எப்படி இருக்கிறீர்கள்"),
    ("how are you doing", "நீங்கள் எப்படி இருக்கிறீர்கள்"),

    # Common Daily Sentences & Expressions
    ("i am fine", "நான் நலமாக இருக்கிறேன்"),
    ("i am doing good", "நான் நன்றாக இருக்கிறேன்"),
    ("my name is ram", "என் பெயர் ராம்"),
    ("i love tamil", "எனக்கு தமிழ் பிடிக்கும்"),
    ("i speak english", "நான் ஆங்கிலம் பேசுகிறேன்"),
    ("i am learning tamil", "நான் தமிழ் கற்கிறேன்"),
    ("do you speak english", "நீங்கள் ஆங்கிலம் பேசுவீர்களா"),
    ("do you speak tamil", "நீங்கள் தமிழ் பேசுவீர்களா"),
    ("i understand", "எனக்கு புரிகிறது"),
    ("i do not understand", "எனக்கு புரியவில்லை"),
    ("help me", "எனக்கு உதவுங்கள்"),
    ("can you help me", "எனக்கு உதவ முடியுமா"),
    ("come here", "இங்கே வா"),
    ("go there", "அங்கே போ"),
    ("sit down", "உட்காருங்கள்"),
    ("stand up", "எழுந்து நில்லுங்கள்"),
    ("open the door", "கதவை திறக்கவும்"),
    ("close the door", "கதவை மூடவும்"),
    ("turn on the light", "விளக்கை போடுங்கள்"),
    ("turn off the light", "விளக்கை அணைக்கவும்"),
    ("listen to me", "நான் சொல்வதை கேளுங்கள்"),
    ("read this book", "இந்த புத்தகத்தை படியுங்கள்"),
    ("write your name", "உங்கள் பெயரை எழுதுங்கள்"),

    # Food & Daily Activities
    ("water", "தண்ணீர்"),
    ("food", "உணவு"),
    ("milk", "பால்"),
    ("tea", "தேநீர்"),
    ("coffee", "காபி"),
    ("rice", "சாதம்"),
    ("fruit", "பழம்"),
    ("apple", "ஆப்பிள்"),
    ("mango", "மாம்பழம்"),
    ("banana", "வாழைப்பழம்"),
    ("i am hungry", "எனக்கு பசிக்கிறது"),
    ("i am thirsty", "எனக்கு தாகமாக இருக்கிறது"),
    ("i want water", "எனக்கு தண்ணீர் வேண்டும்"),
    ("i want food", "எனக்கு உணவு வேண்டும்"),
    ("drink water", "தண்ணீர் குடியுங்கள்"),
    ("eat food", "உணவு சாப்பிடுங்கள்"),
    ("have your breakfast", "காலை உணவு சாப்பிடுங்கள்"),
    ("have your lunch", "மதிய உணவு சாப்பிடுங்கள்"),
    ("have your dinner", "இரவு உணவு சாப்பிடுங்கள்"),

    # Family & People
    ("father", "அப்பா"),
    ("mother", "அம்மா"),
    ("brother", "சகோதரன்"),
    ("sister", "சகோதரி"),
    ("son", "மகன்"),
    ("daughter", "மகள்"),
    ("grandfather", "தாத்தா"),
    ("grandmother", "பாட்டி"),
    ("friend", "நண்பன்"),
    ("teacher", "ஆசிரியர்"),
    ("student", "மாணவன்"),
    ("doctor", "மருத்துவர்"),
    ("this is my father", "இவர் என் தந்தை"),
    ("this is my mother", "இவர் என் தாய்"),
    ("he is my friend", "அவன் என் நண்பன்"),
    ("she is my sister", "அவள் என் சகோதரி"),
    ("they are my friends", "அவர்கள் என் நண்பர்கள்"),

    # Places, Nature & Objects
    ("home", "வீடு"),
    ("house", "வீடு"),
    ("school", "பள்ளி"),
    ("college", "கல்லூரி"),
    ("office", "அலுவலகம்"),
    ("sun", "சூரியன்"),
    ("moon", "நிலா"),
    ("star", "நட்சத்திரம்"),
    ("sky", "வானம்"),
    ("rain", "மழை"),
    ("flower", "மலர்"),
    ("tree", "மரம்"),
    ("book", "புத்தகம்"),
    ("pen", "பேனா"),
    ("pencil", "பென்சில்"),
    ("car", "மகிழுந்து"),
    ("bus", "பேருந்து"),
    ("train", "ரயில்"),
    ("it is raining", "மழை பெய்கிறது"),
    ("the sun rises", "சூரியன் உதிக்கிறது"),
    ("the sky is blue", "வானம் நீல நிறமாக உள்ளது"),
    ("this book is good", "இந்த புத்தகம் நன்றாக உள்ளது"),

    # Numbers & Time
    ("one", "ஒன்று"),
    ("two", "இரண்டு"),
    ("three", "மூன்று"),
    ("four", "நான்கு"),
    ("five", "ஐந்து"),
    ("six", "ஆறு"),
    ("seven", "ஏழு"),
    ("eight", "எட்டு"),
    ("nine", "ஒன்பது"),
    ("ten", "பத்து"),
    ("today", "இன்று"),
    ("tomorrow", "நாளை"),
    ("yesterday", "நேற்று"),
    ("now", "இப்போது"),
    ("morning", "காலை"),
    ("afternoon", "மதியம்"),
    ("evening", "மாலை"),
    ("night", "இரவு"),
    ("what is the time now", "இப்போது மணி என்ன"),
    ("today is a good day", "இன்று ஒரு நல்ல நாள்"),
    ("i will come tomorrow", "நான் நாளை வருவேன்"),
    ("he came yesterday", "அவன் நேற்று வந்தான்"),

    # Feelings, Adjectives & Common Sentences
    ("happy", "மகிழ்ச்சி"),
    ("sad", "வருத்தம்"),
    ("big", "பெரிய"),
    ("small", "சிறிய"),
    ("hot", "சூடான"),
    ("cold", "குளிர்ந்த"),
    ("fast", "வேகமாக"),
    ("slow", "மெதுவாக"),
    ("beautiful", "அழகான"),
    ("i am happy", "நான் மகிழ்ச்சியாக இருக்கிறேன்"),
    ("she is beautiful", "அவள் அழகாக இருக்கிறாள்"),
    ("this is very big", "இது மிகவும் பெரியது"),
    ("this is very small", "இது மிகவும் சிறியது"),
    ("water is cold", "தண்ணீர் குளிர்ச்சியாக உள்ளது"),
    ("tea is hot", "தேநீர் சூடாக இருக்கிறது"),
    ("he runs fast", "அவன் வேகமாக ஓடுகிறான்"),
    ("walk slowly", "மெதுவாக நடங்கள்"),
    ("speak slowly", "மெதுவாக பேசுங்கள்"),
    ("i am going to school", "நான் பள்ளிக்கு செல்கிறேன்"),
    ("i am going home", "நான் வீட்டுக்கு போகிறேன்"),
    ("we are going home", "நாங்கள் வீட்டுக்கு போகிறோம்"),
    ("they are playing", "அவர்கள் விளையாடுகிறார்கள்"),
    ("the baby is sleeping", "குழந்தை தூங்குகிறது"),
    ("i want to sleep", "நான் தூங்க விரும்புகிறேன்"),
    ("let us go", "நாம் செல்வோம்"),
    ("all the best", "வாழ்த்துகள்"),
    ("congratulations", "வாழ்த்துகள்"),
    ("take care", "பார்த்துக்கொள்ளுங்கள்")
]


def load_dataset(file_path=None):
    """
    Loads English-Tamil parallel pairs.
    If file_path is provided (TSV or CSV with 'english\ttamil' or 'english,tamil'),
    it loads from file; otherwise uses the curated bilingual data.
    """
    pairs = list(DEFAULT_PARALLEL_DATA)

    if file_path and os.path.exists(file_path):
        print(f"Loading external dataset from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    parts = line.split("\t")
                elif "," in line:
                    parts = line.split(",", 1)
                else:
                    continue
                if len(parts) >= 2:
                    en = parts[0].strip()
                    ta = parts[1].strip()
                    if en and ta:
                        pairs.append((en, ta))

    # Basic normalization
    cleaned_pairs = []
    for en, ta in pairs:
        en_clean = en.strip().lower()
        ta_clean = ta.strip()
        if en_clean and ta_clean:
            cleaned_pairs.append((en_clean, ta_clean))

    # Remove duplicates preserving order
    seen = set()
    unique_pairs = []
    for en, ta in cleaned_pairs:
        key = (en, ta)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((en, ta))

    print(f"Total dataset pairs loaded: {len(unique_pairs)}")
    return unique_pairs
