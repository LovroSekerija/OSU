list = []
count = 0
while 1:
    x = input("Enter a number or Done:")
    if(x == "Done"):
        break
    try:
        x = float(x)
        list.append(x)
        count +=1
    except ValueError:
        print("Try again")

if count > 0:
    average = sum(list) / count
    minimum = min(list)
    maximum = max(list)
    list.sort

    print(f"Broj unesenih brojeva: {count}")
    print(f"Srednja vrijednost: {average}")
    print(f"Minimalna vrijednost: {minimum}")
    print(f"Maksimalna vrijednost: {maximum}")
    print(f"Sortirana lista: {list}")
else:
    print("List is empyt")