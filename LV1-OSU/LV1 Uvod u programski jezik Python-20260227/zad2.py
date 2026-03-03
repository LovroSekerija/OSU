while 1:
    try:
        x = input("Enter number in [0.0, 1.0]:")
        if (x == ""):
            raise ValueError("You sholud eneter number")
        x = float(x)
        if( x < 0.0 or x > 1.0):
            raise ValueError("Not in range")
        elif( x >= 0.9):
            print("A")
        elif( x < 0.9 and x >= 0.8):
            print("B")
        elif( x < 0.8 and x >= 0.7):
            print("C")
        elif( x < 0.7 and x >= 0.6):
            print("D")
        else:
            print("F")
    except ValueError as e:
        print("Error:", e)
