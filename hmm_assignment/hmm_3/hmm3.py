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

    # Combine the last two lines of data into a single list and convert it to a NumPy array
    emission_sequence = np.array(data[3].split(' ')[1:], dtype=int)

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

    emission_sequence = [int(x) for x in input().split()]
    emission_sequence = np.array(emission_sequence[1:], dtype=int)

    return transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence


def createMatrix(data, no_of_rows, no_of_columns):
    matrix = np.zeros((no_of_rows, no_of_columns))
    index = 2
    for i in range(0, no_of_rows):
        for j in range(0, no_of_columns):
            matrix[i][j] = data[index]
            index += 1
    return matrix


def estimate_model(transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence):
    T = len(emission_sequence)
    N = len(transition_matrix)
    M = len(emission_matrix[0])



    # Forward algorithm to calculate probabilities of each state at each time step
    alpha = np.zeros((T, N))
    for t in range(T):
        for j in range(N):
            if t == 0:
                alpha[t, j] = initial_state_probability_matrix[0, j] * emission_matrix[j, emission_sequence[t]]
            else:
                alpha[t, j] = emission_matrix[j, emission_sequence[t]] * sum(
                    alpha[t - 1, i] * transition_matrix[i, j] for i in range(N)
                )

    # Backward algorithm to calculate probabilities of each state at each time step
    beta = np.zeros((T, N))
    beta[-1, :] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(
                transition_matrix[i, :] * emission_matrix[:, emission_sequence[t + 1]] * beta[t + 1, :]
            )

    # Estimate new transition matrix
    new_transition_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            numerator = sum(
                alpha[t, i] * transition_matrix[i, j] * emission_matrix[j, emission_sequence[t + 1]] * beta[t + 1, j]
                for t in range(T - 1)
            )
            denominator = sum(alpha[t, i] * beta[t, i] for t in range(T - 1))

            # Avoid division by zero
            if denominator != 0:
                new_transition_matrix[i, j] = numerator / denominator
            else:
                new_transition_matrix[i, j] = 0

    # Estimate new emission matrix
    new_emission_matrix = np.zeros((N, M))
    for j in range(N):
        for k in range(M):
            numerator = sum(alpha[t, j] * (emission_sequence[t] == k) * beta[t, j] for t in range(T))
            denominator = sum(alpha[t, j] * beta[t, j] for t in range(T))

            # Avoid division by zero
            if denominator != 0:
                new_emission_matrix[j, k] = numerator / denominator
            else:
                new_emission_matrix[j, k] = 0

    return new_transition_matrix, new_emission_matrix


def main():
    data = readData('data.txt')
    transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence = readMatrix(data)

    new_transition_matrix, new_emission_matrix = estimate_model(
        transition_matrix, emission_matrix, initial_state_probability_matrix, emission_sequence
    )

    # Print the estimated transition matrix
    print(f"{len(new_transition_matrix)} {len(new_transition_matrix[0])}")
    for row in new_transition_matrix:
        print(" ".join(map(str, row)))

    # Print the estimated emission matrix
    print(f"{len(new_emission_matrix)} {len(new_emission_matrix[0])}")
    for row in new_emission_matrix:
        print(" ".join(map(str, row)))


if __name__ == "__main__":
    main()
