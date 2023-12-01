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

    # Convert emission_sequence from string to list of integers
    emission_sequence = [int(x) for x in data[3].split(' ')[1:]]

    return transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence


def readInputKattis():
    transition_data = [float(x) for x in input().split()]
    transition_matrix = createMatrix(transition_data, int(transition_data[0]), int(transition_data[1]))

    emission_data = [float(x) for x in input().split()]
    emission_matrix = createMatrix(emission_data, int(emission_data[0]), int(emission_data[1]))

    initial_state_probability_data = [float(x) for x in input().split()]
    initial_state_probability_matrix = createMatrix(initial_state_probability_data,
                                                    int(initial_state_probability_data[0]),
                                                    int(initial_state_probability_data[1]))

    # Convert emission_sequence from string to list of integers
    emission_sequence = [int(x) for x in input().split()][1:]

    return transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence


def createMatrix(data, no_of_rows, no_of_columns):
    matrix = [[0 for _ in range(no_of_columns)] for _ in range(no_of_rows)]
    index = 2
    for i in range(no_of_rows):
        for j in range(no_of_columns):
            matrix[i][j] = data[index]
            index += 1
    return matrix


def viterbi_algorithm(transition_matrix, emission_matrix, initial_state_probability_matrix, emissions_sequence):
    T = len(emissions_sequence)
    N = len(transition_matrix)

    # Initialization
    V = [[0 for _ in range(N)] for _ in range(T)]
    BP = [[0 for _ in range(N)] for _ in range(T)]

    for i in range(N):
        V[0][i] = initial_state_probability_matrix[0][i] * emission_matrix[i][emissions_sequence[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            prob_states = [V[t - 1][i] * transition_matrix[i][j] * emission_matrix[j][emissions_sequence[t]] for i in range(N)]
            BP[t][j] = prob_states.index(max(prob_states))
            V[t][j] = max(prob_states)

    # Termination
    best_last_state = V[T - 1].index(max(V[T - 1]))

    # Backtrack
    best_path = [0 for _ in range(T)]
    best_path[T - 1] = best_last_state

    for t in range(T - 2, -1, -1):
        best_path[t] = BP[t + 1][best_path[t + 1]]

    return best_path



def main():
    # data = readData('data.txt')
    # transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence = readMatrix(data)
    transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence = readInputKattis()

    ans = viterbi_algorithm(transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence)
    print(" ".join(map(str, ans)))  # Output as space-separated string

if __name__ == "__main__":
    main()
