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
    matrix = [[0 for _ in range(no_of_columns)] for _ in range(no_of_rows)]
    index = 2
    for i in range(no_of_rows):
        for j in range(no_of_columns):
            matrix[i][j] = data[index]
            index += 1
    return matrix

def matrixMultiplication(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def main():
    # data = readData('data.txt')
    # transition_matrix, emission_matrix, initial_state_probability_matrix = readMatrix(data)

    transition_matrix, emission_matrix, initial_state_probability_matrix = readInputKattis()

    first_transition_matrix = matrixMultiplication(initial_state_probability_matrix, transition_matrix)
    probability_matrix = matrixMultiplication(first_transition_matrix, emission_matrix)

    result = [len(probability_matrix), len(probability_matrix[0])]
    result.extend(sum(probability_matrix, []))
    # round off all values of result to 2 decimal places
    result = [round(elem, 2) for elem in result]
    # convert all elements of result to string for kattis
    result = ' '.join(map(str, result))
    print(result)


if __name__ == "__main__":
    main()
