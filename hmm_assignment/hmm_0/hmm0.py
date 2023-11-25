import numpy as np


def readData(file_name):
    with open(file_name, 'r') as f:
        data = f.read().splitlines()
    return data


def readMatrix(data):
    transition_data = data[0].split(' ')
    transition_matrix = createMatrix(transition_data, int(transition_data[0]), int(transition_data[1]))
    emission_data = data[1].split(' ')
    emission_matrix = createMatrix(emission_data, int(emission_data[0]), int(emission_data[1]))
    initial_state_probability_data = data[2].split(' ')
    initial_state_probability_matrix = createMatrix(initial_state_probability_data,
                                                    int(initial_state_probability_data[0]),
                                                    int(initial_state_probability_data[1]))
    return transition_matrix, emission_matrix, initial_state_probability_matrix


def readInputKattis():
    transition_data = []
    emission_data = []
    initial_state_probability_data = []
    for x in input().split():
        transition_data.append(float(x))
    transition_matrix = createMatrix(transition_data, int(transition_data[0]), int(transition_data[1]))
    for x in input().split():
        emission_data.append(float(x))
    emission_matrix = createMatrix(emission_data, int(emission_data[0]), int(emission_data[1]))
    for x in input().split():
        initial_state_probability_data.append(float(x))
    initial_state_probability_matrix = createMatrix(initial_state_probability_data,
                                                    int(initial_state_probability_data[0]),
                                                    int(initial_state_probability_data[1]))
    return transition_matrix, emission_matrix, initial_state_probability_matrix


def createMatrix(data, no_of_rows, no_of_columns):
    matrix = np.zeros((no_of_rows, no_of_columns))
    index = 2
    for i in range(0, no_of_rows):
        for j in range(0, no_of_columns):
            matrix[i][j] = data[index]
            index += 1
    return matrix


def main():
    # data = readData('data.txt')
    # transition_matrix, emission_matrix, initial_state_probability_matrix = readMatrix(data)

    transition_matrix, emission_matrix, initial_state_probability_matrix = readInputKattis()

    first_transition_matrix = np.dot(initial_state_probability_matrix, transition_matrix)
    probability_matrix = np.dot(first_transition_matrix, emission_matrix)

    result = [probability_matrix.shape[0], probability_matrix.shape[1]]
    result.extend(probability_matrix[0].tolist())
    # round off all values of result to 2 decimal places
    result = [round(elem, 2) for elem in result]
    # convert all elements of result to string for kattis
    result = ' '.join(map(str, result))
    print(result)


if __name__ == "__main__":
    main()
