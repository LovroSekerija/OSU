file = open("song.txt", "r", encoding="utf-8")
wordsDict = {}

for line in file:
    line = line.strip()
    words = line.split()
    for word in words:
        word = word.lower().strip(".,!?;:\"()")
        if word in wordsDict:
            wordsDict[word] += 1
        else:
            wordsDict[word] = 1

file.close()

onlyWord = []

for word, number in wordsDict.items():
    if number == 1:
        onlyWord.append(word)
print(len(onlyWord))
for word in onlyWord:
    print(word)