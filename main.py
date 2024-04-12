import csv
import numpy as np
from scale import scale_into_number


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

    data = np.array(list1)
    label = np.array(list2)
    return data, label


if __name__ == '__main__':
    reading_files("titanic.csv")
