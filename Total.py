import re
import gzip
import numpy as np
import scipy.spatial.distance
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class Model:
    def __init__(player, model="vectors_german.txt.gz", dictionary="vocab_german.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        player.model_file = model
        player.dictionary_file = dictionary
        player.pattern = re.compile(pattern)
        player.words = set()

    def load_words(player):
        print("Loading words from dictionary...")
        with open(player.dictionary_file, "r", encoding="utf-8") as f:
            player.words.update(line.strip() for line in f if player.pattern.match(line))
        print(f"Loaded {len(player.words)} words.")

    def get_vector(player, word):
        with gzip.open(player.model_file, "rt", encoding="utf-8") as f:
            for line in f:
                tokens = line.split(" ")
                if tokens[0] == word:
                    return np.array(tokens[1:], dtype=np.float32)
        return None

    def distance(player, word1, word2):
        vector1 = player.get_vector(word1)
        vector2 = player.get_vector(word2)
        if vector1 is not None and vector2 is not None:
            return scipy.spatial.distance.cosine(vector1, vector2) * 100
        return None

    def calculate_originality(player, word_pair, mystery_word):
        word1, word2 = word_pair.split(" + ")
        print(f"Calculating distance between '{word1}' and '{mystery_word}'...")
        dist_1 = player.distance(word1, mystery_word)
        print(f"Distance between '{word1}' and '{mystery_word}': {dist_1}")
        print(f"Calculating distance between '{word2}' and '{mystery_word}'...")
        dist_2 = player.distance(word2, mystery_word)
        print(f"Distance between '{word2}' and '{mystery_word}': {dist_2}")
        if dist_1 is not None and dist_2 is not None:
            originality = (dist_1 + dist_2) / 2
            print(f"Originality for '{word_pair}' and '{mystery_word}': {originality}")
            return originality
        else:
            print(f"Could not calculate originality for '{word_pair}' and '{mystery_word}'.")
            return None

    def __enter__(player):
        print("Entering Model context...")
        player.load_words()
        return player

    def __exit__(player, exc_type, exc_value, traceback):
        print("Exiting Model context.")

# Model instanziieren
print("Creating Model instance...")
model = Model("vectors_german.txt.gz", "vocab_german.txt")

# Mystery-Wörter
mystery_words = ['raum', 'taube', 'golf', 'elektrizität', 'ende', 'sombrero']

# CSV-Datei einlesen und Umlaute ersetzen
print("Reading CSV file...")
df = pd.read_csv('pre_test_combined_edit.csv', delimiter=',', encoding='utf-8')

# Ersetze ae, oe, ue in allen Vote-Spalten
print("Replacing umlauts in vote columns...")
vote_columns = df.columns[1:]
df[vote_columns] = df[vote_columns].replace({'ae': 'ä', 'oe': 'ö',  r'(?<!q)ue': 'ü' }, regex=True) 


# Liste, um die Originalitätswerte zu speichern
originality_measures = {i: [] for i in range(1, 7)}

# Funktion zur Berechnung der Originalität für eine Runde und ein Mystery-Wort
def calculate_originality_for_round(model, row, round_number, mystery_word):
    word_pair = row[f'new_justone{round_number}playervote_group']
    print(f"Calculating originality for round {round_number} and mystery word '{mystery_word}' with word pair '{word_pair}'...")
    originality = model.calculate_originality(word_pair, mystery_word)
    return originality

# Calculate originality measures for each round and mystery word
print("Calculating originality measures...")
with model as model_instance:
    with ThreadPoolExecutor() as executor:
        for round_number in range(1, 7):
            print(f"=== Round {round_number} ===")
            for idx, row in df.iterrows():
                mystery_word = mystery_words[round_number - 1]  # Select the corresponding mystery word for this round
                print(f"Processing row {idx}...")
                originality = calculate_originality_for_round(model_instance, row, round_number, mystery_word)
                originality_measures[round_number].append(originality)

# Add originality values to the DataFrame
print("Adding originality values to DataFrame...")
for round_number in range(1, 7):
    df[f'originality_{round_number}'] = originality_measures[round_number]

# Save the DataFrame with originality values to a new CSV file
print("Saving DataFrame to CSV...")
df.to_csv('pre_test_combined_edit_with_originality.csv', index=False, sep=',', encoding='utf-8')

print("Originality calculation completed and saved to CSV.")
