import re
import gzip
import numpy as np
import scipy.spatial.distance
from concurrent.futures import ThreadPoolExecutor

class Model:
    def __init__(player, model="vectors_german.txt.gz", dictionary="vocab_german.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        player.model_file = model
        player.dictionary_file = dictionary
        player.pattern = re.compile(pattern)
        player.words = set()

    def load_words(player):
        with open(player.dictionary_file, "r", encoding="utf-8") as f:
            player.words.update(line.strip() for line in f if player.pattern.match(line))

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
        dist_1 = player.distance(word_pair.split(" + ")[0], mystery_word)
        dist_2 = player.distance(word_pair.split(" + ")[1], mystery_word)
        print(dist_1, dist_2)
        if dist_1 is not None and dist_2 is not None:
            return (dist_1 + dist_2) / 2
        return None
    
    def __enter__(player):
        player.load_words()
        return player

    def __exit__(player, exc_type, exc_value, traceback):
        pass

# Example:
word_pairs = [
    "tacos + sonne", "kappe + mittelamerika", "mütze + tequila", "kleidung + zentralamerika", "tequila + mütze", "tequila + deckel", "lateinamerika + accessoire", "kappe + mittelamerika", "spanisch + mütze", "lateinamerika + mütze"]
mystery_word = "sombrero"

with Model() as model:
    with ThreadPoolExecutor() as executor:
        originality_measures = []
        for pair in word_pairs:
            originality_future = executor.submit(model.calculate_originality, pair, mystery_word)
            originality_measures.append(originality_future.result())

    for idx, originality in enumerate(originality_measures, start=1):
        if originality is not None:
            print(f"Paar {idx}: {originality}")
        else:
            print(f"Originalitätsmaß für Paar {idx} nicht berechenbar")