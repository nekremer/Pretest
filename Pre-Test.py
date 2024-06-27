import dat
import csv

model = dat.Model("glove.840B.300d.txt", "words.txt") # load model   

# Öffnen der CSV-Datei und Lesen der Daten
with open('DAT_Data.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Überspringen der Kopfzeile mit den Spaltennamen

    word_lists = []
    DAT_scores = []

    for row in reader:
        words_for_id = [word.strip() for word in row[1:]]
        word_lists.append(words_for_id)

for i, words in enumerate(word_lists, 1):
    print(words)
    DAT_score = model.dat(words) 
    DAT_scores.append(DAT_score)
    print(DAT_score)


# Öffnen der CSV-Datei im Schreibmodus, um die neuen Daten zu speichern
with open('DAT_Data_Score.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Schreiben der Kopfzeile mit den Spaltennamen
    writer.writerow(['id', 'word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10', 'DAT_score'])

    # Schreiben der Daten für jede ID und deren zugehörigen Zahlenwert in die CSV-Datei
    for i, (words, score) in enumerate(zip(word_lists, DAT_scores), 1):
        writer.writerow([i] + words + [score])
