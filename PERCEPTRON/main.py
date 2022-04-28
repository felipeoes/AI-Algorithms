from file_manager import FileManager
from perceptron import Perceptron
import random

# PROBLEM AND
file_manager = FileManager("data/problemAND.csv")
df_AND = file_manager.read_csv(
    cols_names=["expression1", "expression2", "result"])
X, y = file_manager.extract_X_y(df_AND)

print(f"X: {X} \n")
print(f"y: {y} \n")

# Input layer
weights = {
    col_num: random.random() for col_num in range(len(df_AND.columns) - 1)
}
bias = random.random()

perceptron = Perceptron(weights, bias)

learning_rates = [0.01, 0.05, 0.1]
learning_rate = random.choice(learning_rates)

# Processing and output layer
accuracy = perceptron.train(X, y, epochs=50, learning_rate=learning_rate)
print("Mean training accuracy of model: ", accuracy)

params_dict = {"weights": perceptron.weights,
               "bias": perceptron.bias, "learning_rate": learning_rate}
print(f"Params: {params_dict}")

# Testing
predicted = perceptron.predict(X)
print(f"Predicted: {predicted}")

predicted_0 = perceptron.predict([[1, 1]])
print(f"Predicted 0: {predicted_0}")

# CHARS PROBLEM
chars_dict = {
    '[1, -1, -1, -1, -1, -1, -1]': "A",
    '[-1, 1, -1, -1, -1, -1, -1]': "B",
    '[-1, -1, 1, -1, -1, -1, -1]': "C",
    '[-1, -1, -1, 1, -1, -1, -1]': "D",
    '[-1, -1, -1, -1, 1, -1, -1]': "E",
    '[-1, -1, -1, -1, -1, 1, -1]': "J",
    '[-1, -1, -1, -1, -1, -1, 1]': "K"

}

file_manager = FileManager("data/caracteres-limpo.csv")
df_chars = file_manager.read_csv(chars_df=True)
X, y = file_manager.extract_X_y(df_chars, chars_df=True)

print(f"X: {X} \n")
print(f"y: {y} \n")

learning_rates = [0.01, 0.05]
learning_rate = random.choice(learning_rates)

# A
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronA = Perceptron(weights, bias)
accuracyA = perceptronA.train(
    X, y, epochs=100, learning_rate=learning_rate, chars_df=True, char_index=0)

# B
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronB = Perceptron(weights, bias)
accuracyB = perceptronB.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=1)


# C
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronC = Perceptron(weights, bias)
accuracyC = perceptronC.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=2)

# D
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronD = Perceptron(weights, bias)
accuracyD = perceptronD.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=3)

# E
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronE = Perceptron(weights, bias)
accuracyE = perceptronE.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=4)

# J
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronJ = Perceptron(weights, bias)
accuracyJ = perceptronJ.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=5)

# K
weights = {col_num: random.random()
           for col_num in range(len(df_chars.columns) - 7)}
bias = random.random()
perceptronK = Perceptron(weights, bias)
accuracyK = perceptronK.train(
    X, y, epochs=50, learning_rate=learning_rate, chars_df=True, char_index=6)


network = [perceptronA, perceptronB, perceptronC,
           perceptronD, perceptronE, perceptronJ, perceptronK]

# Testing

file_manager = FileManager("data/caracteres_teste.csv")
df_char_test = file_manager.read_csv(chars_df=True)
X_test, _ = file_manager.extract_X_y(df_char_test, chars_df=True)

final_predict = []
for index, neuron in enumerate(network):
    print(f"Bias neuron {index} : {neuron.bias} \n")

    predicted = neuron.predict([*X_test], value_only=True)
    final_predict.append([predicted[index] for index in predicted])

for index, prediction in enumerate(final_predict):
    letter = [final_predict[index][num] for num in range(len(final_predict))]
    if str(letter) == list(chars_dict)[index]:
        char = chars_dict[str(letter)]
        print(
            f"Predicted the letter {char} correctly with accuracy {globals()[f'accuracy{char}']}! Generated vector: {letter}")
    else:
        try: 
            char = chars_dict[str(letter)]
        except:
            char = "Unknown"
        print(
            f"Predicted the letter {char} incorrectly! | Generated vector: {letter}")
