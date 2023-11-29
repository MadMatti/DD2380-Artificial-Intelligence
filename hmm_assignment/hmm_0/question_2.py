import numpy as np

# Given matrices
A = np.array([[0.5, 0.5], [0.5, 0.5]])
v = np.array([0.5, 0.5])
B = np.array([[0.9, 0.1], [0.5, 0.5]])

# Compute the result of multiplying the current state distribution vector by the transition matrix
v_prime = np.dot(v, A)

# Compute the result of multiplying v_prime by the observation matrix
next_observation_distribution = np.dot(v_prime, B)

print("Transition matrix A:")
print(A)

print("\nCurrent state distribution vector v:")
print(v)

print("\nResult of multiplying v by A (v_prime):")
print(v_prime)

print("\nObservation matrix B:")
print(B)

print("\nNext Observation Distribution:")
print(next_observation_distribution)
