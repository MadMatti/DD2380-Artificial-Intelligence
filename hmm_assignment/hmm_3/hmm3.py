import math

# Reads user input
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
    pi_matrix = createMatrix(initial_state_probability_data,
                              int(initial_state_probability_data[0]),
                              int(initial_state_probability_data[1]))

    # Combine the last two lines of data into a single list
    emission_sequence = [int(x) for x in data[3].split(' ')[1:]]

    return transition_matrix, emission_matrix, pi_matrix, emission_sequence

def readInputKattis():
    transition_data = [float(x) for x in input().split()]
    transition_matrix = createMatrix(transition_data, int(transition_data[0]), int(transition_data[1]))

    emission_data = [float(x) for x in input().split()]
    emission_matrix = createMatrix(emission_data, int(emission_data[0]), int(emission_data[1]))

    initial_state_probability_data = [float(x) for x in input().split()]
    pi_matrix = createMatrix(initial_state_probability_data,
                              int(initial_state_probability_data[0]),
                              int(initial_state_probability_data[1]))

    emission_sequence = [int(x) for x in input().split()[1:]]  # Exclude the first element
    return transition_matrix, emission_matrix, pi_matrix, emission_sequence


def createMatrix(data, no_of_rows, no_of_columns):
    matrix = []
    index = 2
    for i in range(no_of_rows):
        row = []
        for j in range(no_of_columns):
            row.append(data[index])
            index += 1
        matrix.append(row)
    return matrix


def alpha_pass(transition_matrix, emission_matrix, pi_matrix, emission_sequence):
    T = len(emission_sequence)
    N = len(transition_matrix)

    # Create alpha matrix and scaling factor for each observation
    alpha = [[0 for x in range(T)]
             for y in range(N)]
    scale = [0 for x in range(T)]

    # Initialize alpha matrix with the first observation
    for state in range(N):
        alpha[state][0] = pi_matrix[0][state] * \
            emission_matrix[state][emission_sequence[0]]
        # Calculate the scaling factor --> sum of all alpha values for each observation
        scale[0] += alpha[state][0]

    scale[0] = (1/scale[0])
    for state in range(N):
        alpha[state][0] = scale[0] * alpha[state][0]

    # Calculate alpha matrix for all observations
    for observation in range(1, T):
        for to_state in range(N):
            # Calculate the scaling factor --> sum of all alpha values for each observation
            for from_state in range(N):
                # Calculate the probability of being in state i at time t given the observations up to the current observation
                alpha[to_state][observation] += alpha[from_state][observation -
                                                                  1] * transition_matrix[from_state][to_state]
            # Multiply with the probability of observing o_t given that we are in state i
            alpha[to_state][observation] *= emission_matrix[to_state][emission_sequence[observation]]
            scale[observation] += alpha[to_state][observation]

        scale[observation] = 1 / scale[observation]
        for state in range(N):
            alpha[state][observation] = scale[observation] * \
                alpha[state][observation]

    return alpha, scale

def beta_pass(transition_matrix, emission_matrix, emission_sequence, scale):
    T = len(emission_sequence)
    N = len(transition_matrix)

    # Create beta matrix
    beta = [[0 for x in range(T)]
            for y in range(N)]

    # Initialize beta matrix with the last observation
    for state in range(N):
        beta[state][T-1] = scale[T-1]

    # Calculate beta matrix for all observations
    for observation in range(T-2, -1, -1):
        for from_state in range(N):
            beta[from_state][observation] = 0
            for to_state in range(N):
                beta[from_state][observation] += transition_matrix[from_state][to_state] * \
                    emission_matrix[to_state][emission_sequence[observation+1]
                                              ] * beta[to_state][observation+1]
            beta[from_state][observation] = scale[observation] * \
                beta[from_state][observation]
    return beta

def calculate_gamma(alpha, beta, transition_matrix, emission_matrix, emission_sequence):
    total_observation = len(emission_sequence)
    N = len(transition_matrix)

    # Create gamma and di_gamma matrix
    gamma = [[0 for x in range(total_observation)]
             for y in range(N)]
    di_gamma = [[[0 for x in range(total_observation)] for y in range(
        N)] for z in range(N)]

    # Calculate gamma and di_gamma matrix
    for observation in range(total_observation-1):
        for from_state in range(N):
            for to_state in range(N):
                # Calculate the probability of being in state i at time t and transitioning to state j at time t+1 given all observations
                di_gamma[to_state][from_state][observation] = alpha[from_state][observation] * transition_matrix[from_state][to_state] * \
                    emission_matrix[to_state][emission_sequence[observation+1]
                                              ] * beta[to_state][observation+1]
                # Calculate the probability of being in state i at time t given the observations coming after, and all the observations coming before
                gamma[from_state][observation] += di_gamma[to_state][from_state][observation]
                # Marginalizing out the destination variable from di-gamma
    for state in range(N):
        gamma[state][total_observation-1] = alpha[state][total_observation-1]
    return gamma, di_gamma

def re_estimate(gamma, di_gamma, transition_matrix, emission_matrix, pi_matrix, emission_sequence):
    total_observation = len(emission_sequence)
    N = len(transition_matrix)

    # Re-estimate the initial state probabilities
    for state in range(N):  # for t = 0
        pi_matrix[0][state] = gamma[state][0]

    # Re-estimate the transition probabilities
    for from_state in range(N):
        denominator = 0
        for observation in range(total_observation-1):
            # Calculate the denominator for the transition probabilities
            # expected number of transitions from i to any state
            denominator += gamma[from_state][observation]
        for to_state in range(N):
            numerator = 0
            for observation in range(total_observation-1):
                # Calculate the numerator for the transition probabilities
                # Expected number of transitions from state i to j
                numerator += di_gamma[to_state][from_state][observation]
            # Calculate the transition probabilities
            # Probability of moving from state i to j
            transition_matrix[from_state][to_state] = numerator / denominator

    # Re-estimate the emission probabilities
    for from_state in range(N):
        denominator = 0
        for observation in range(total_observation):
            # Calculate the denominator for the emission probabilities
            # expected number of times the model is in state i
            denominator += gamma[from_state][observation]
        for emission in range(len(emission_matrix[0])):
            numerator = 0
            for observation in range(total_observation):
                # Check if the emission is the same as the observation
                if emission_sequence[observation] == emission:
                    # expected number of times the model is in state i with observation k
                    numerator += gamma[from_state][observation]
            # Calculate the emission probabilities
            # Probability of observing a certain observation K given state i.
            emission_matrix[from_state][emission] = numerator / denominator

    return transition_matrix, emission_matrix, pi_matrix

def log_likelihood(scale):
    total_observation = len(scale)
    log_prob = 0
    for observation in range(total_observation):
        # Calculate the probability of seeing a sequence of observations
        log_prob += math.log(scale[observation])
    log_prob = -log_prob
    return log_prob

def baum_welch_algorithm(transition_matrix, emission_matrix, pi_matrix, emission_sequence, max_iterations):
    log_prob = -math.inf
    alpha, scaling_factor = alpha_pass(
        transition_matrix, emission_matrix, pi_matrix, emission_sequence)

    for iteration in range(max_iterations):
        # Calculate the alpha, beta, and gamma matrix
        alpha, scaling_factor = alpha_pass(
            transition_matrix, emission_matrix, pi_matrix, emission_sequence)
        beta = beta_pass(transition_matrix, emission_matrix,
                         emission_sequence, scaling_factor)
        gamma, di_gamma = calculate_gamma(
            alpha, beta, transition_matrix, emission_matrix, emission_sequence)

        # Re-estimate the model parameters
        transition_new, emission_new, pi_new = re_estimate(
            gamma, di_gamma, transition_matrix, emission_matrix, pi_matrix, emission_sequence)

        probability = log_likelihood(scaling_factor)

        if probability > log_prob:
            log_prob = probability
        else:
            print(iteration)
            break

    return transition_new, emission_new

def baum_welch_algorithm_v2(transition_matrix, emission_matrix, pi_matrix, emission_sequence, max_iterations, convergence_threshold):
    log_prob = -math.inf
    for iteration in range(max_iterations):
        # Calculate the alpha, beta, and gamma matrix
        alpha, scaling_factor = alpha_pass(
            transition_matrix, emission_matrix, pi_matrix, emission_sequence)
        beta = beta_pass(transition_matrix, emission_matrix,
                         emission_sequence, scaling_factor)
        gamma, di_gamma = calculate_gamma(
            alpha, beta, transition_matrix, emission_matrix, emission_sequence)

        # Re-estimate the model parameters
        transition_new, emission_new, pi_new = re_estimate(
            gamma, di_gamma, transition_matrix, emission_matrix, pi_matrix, emission_sequence)

        # Calculate log-likelihood for the current iteration
        probability = log_likelihood(scaling_factor)
        if iteration == 0: print("Initial log-likelihood: {}".format(probability))

        # Check for convergence based on the change in log-likelihood
        if abs(probability - log_prob) < convergence_threshold:
            print("Converged after {} iterations.".format(iteration + 1))
            print("Convergence log-likelihood: {}".format(probability))
            break

        log_prob = probability

    return transition_new, emission_new

def answer(matrix):
    print(str(len(matrix)) + ' ' + str(len(matrix[0])), end=' ')
    for row in matrix:
        for element in row:
            print(round(element, 6), end=" ")

def main():
    # data = readData('data.txt')
    # transition_matrix, emission_matrix, pi_matrix,  emission_sequence = readMatrix(data)
    transition_matrix, emission_matrix, pi_matrix, emission_sequence = readInputKattis()

    transition_output, emission_output = baum_welch_algorithm_v2(
        transition_matrix, emission_matrix, pi_matrix, emission_sequence, 1000000000, 0.00001)

    # Print the output
    answer(transition_output)
    print()
    answer(emission_output)

if __name__ == '__main__':
    main()
