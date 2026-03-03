file = open("SMSSpamCollection.txt", "r", encoding="utf-8")

ham_poruke = []
spam_poruke = []

for line in file:
    line = line.strip()

    if line.startswith("ham\t"):
        message = line[4:].strip()
        ham_poruke.append(message)
    elif line.startswith("spam\t"):
        message = line[5:].strip()
        spam_poruke.append(message)

averageHam = sum(len(poruka.split()) for poruka in ham_poruke) / len(ham_poruke)
averageSpam = sum(len(poruka.split()) for poruka in spam_poruke) / len(spam_poruke)

print(f"Prosjek riječi u ham porukama: {averageHam:f}")
print(f"Prosjek riječi u spam porukama: {averageSpam:f}")