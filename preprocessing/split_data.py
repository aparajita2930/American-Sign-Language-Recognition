with open('train_complete.txt') as f:
    lines = f.read().splitlines()

lines = lines[:50]
length = len(lines)

validation = 0.1
train = 0.8
test = 1 - validation - train

lenOfValidation = validation * length
lenOfTrain = train * length
lenOfTest  = length - lenOfTrain - lenOfValidation