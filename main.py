import csv
import numpy as np
from scale import scale_into_number
from sklearn.model_selection import train_test_split
from amk_nn import train, test


def reading_files(filename):
    list1 = []
    list2 = []
    file = open(filename)
    csvreader = csv.reader(file)

    for row in csvreader:
        if row[0] == 'PassengerId':
            continue
        scaled = scale_into_number(row)
        list2.append(scaled.pop())
        list1.append(scaled)

    data = np.array(list1, dtype=float)
    label = np.array(list2)
    return data, label


if __name__ == '__main__':
    data, label = reading_files("titanic.csv")
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=43)
    layer_n = 3
    hiddenL_neuron_n = [3]
    train(train_data, train_labels, layer_n, hiddenL_neuron_n, epoch=100, learning_rate=0.1, activation_function=1)

    predictions = test(test_data)
    correct = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            correct += 1
            # print(predictions[i])
    print("Number of correct answers: ", correct)
    print("Number of tests: ", len(test_labels))
    print("Accuracy: ", correct / len(test_labels))
