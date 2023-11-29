#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import numpy as np
import sys
import math

epsilon = sys.float_info.epsilon

def forward_pass(A, B, pi, O,):
    alpha_list = []
    c_list = []

    len_A = len(A)
    len_obs = len(O)

    for t in range(len_obs):
        t_c = 0
        t_alpha_list = []
        t_pi = pi[0]

        for i in range(len_A):
            if t == 0:
                t_alpha = t_pi[i] * B[i][O[t]]
                t_c += t_alpha
                t_alpha_list.append(t_alpha)
            else:
                t_alpha = 0

                for j in range(len_A):
                    t_alpha += alpha_list[t - 1][j] * A[j][i] * B[i][O[t]]
                
                t_c += t_alpha
                t_alpha_list.append(t_alpha)

        for k in range(len_A):
            t_alpha_list[k] = (1 / (t_c + epsilon)) * t_alpha_list[k]
        
        c_list.append(1 / (t_c + epsilon))
        alpha_list.append(t_alpha_list)

    return alpha_list, c_list


def backward_pass(A, B, pi, O, c):
    beta_list = []

    len_A = len(A)
    len_obs = len(O)

    for t in range(len_obs):
        t_beta_list = []

        for i in range(len_A):
            if t == 0:
                t_beta = c[t]
                t_beta_list.append(t_beta)
            else:
                t_sum = 0

                for j in range(len_A):
                    t_sum += beta_list[t-1][j] * A[i][j] * B[j][O[t-1]]
                t_beta_list.append(t_sum)

        if t > 0:
            for k in range(len_A):
                t_beta_list[k] = c[t] * t_beta_list[k]
        
        beta_list.append(t_beta_list)

        return beta_list


def calculate_gamma(A, B, O, alpha_list, beta_list): # seq = O N = len(A), T = len(O)
    gamma_list = []
    di_gamma_list = []

    len_A = len(A)
    len_obs = len(O)

    for t in range(len_obs - 1):
        t_gamma_list = []
        t_di_gamma_list = []

        for i in range(len_A):
            t_gamma_val = []
            gamma = 0

            for j in range(len_A):
                di_gamma = alpha_list[t][i] * A[i][j] * B[j][O[t+1]] * beta_list[t + 1][j]
                gamma += di_gamma
                t_gamma_val.append(di_gamma)
            
            t_gamma_list.append(gamma)
            t_di_gamma_list.append(t_gamma_val)
        
        gamma_list.append(t_gamma_list)
        di_gamma_list.append(t_di_gamma_list)

        t_gamma_list = []

        for k in range(len_A):
            t_gamma_list.append(alpha_list[t+1][k])
        
        gamma_list.append(t_gamma_list)

        return gamma_list, di_gamma_list


def re_estimate(gamma_list, di_gamma_list, O, lenA):
    len_obs = len(O)

    # re estimate pi
    pi_result = []
    for i in range(lenA):
        pi_result.append(gamma_list[0][i])

    # re estimate A
    A_result = []
    for i in range(lenA):
        den = 0
        t_A_list = []

        for t in range(len_obs-1):
            den += gamma_list[t][i]
        
        for j in range(lenA):
            num = 0

            for t in range(len_obs-1):
                t_gamma = di_gamma_list[t][i]
                num += t_gamma[j]

            t_A_list.append(num / (den + epsilon))
        A_result.append(t_A_list)

    # re estimate B
    B_result = []
    for i in range(lenA):
        den = 0
        t_B_list = []

        for t in range(len_obs):
            den += gamma_list[t][i]

        for j in range(len_obs):
            num = 0

            for t in range(len_obs):
                if O[t] == j:
                    num += gamma_list[t][i]

            t_B_list.append(num / (den + epsilon))
        B_result.append(t_B_list)

    return [pi_result], A_result, B_result


def baum_welch(A, B, pi, O):
    len_A = len(A)
    len_obs = len(O)

    iter_cnt = 0
    max_iters = 5
    prev_log_prob = - math.inf
    log_prob = 1

    while iter_cnt < max_iters and log_prob > prev_log_prob:
        iter_cnt += 1
        if iter_cnt != 1:
            prev_log_prob = log_prob

        alpha_values, c_values = forward_pass(A, B, pi, O)

        beta_values = backward_pass(A, B, pi, O[::-1], c_values[::-1])

        gamma_values, di_gamma_values = calculate_gamma(A, B, O, alpha_values, beta_values[::-1])

        pi, A, B, = re_estimate(gamma_values, di_gamma_values, O, len_A)

        log_prob = log_likelihood(c_values, len_obs)

    return A, B, pi


def log_likelihood(c, len_obs):
    log_prob = 0
    for i in range(len_obs):
        log_prob -= np.log(c[i])

    return log_prob

def dot_prod(matrix_a, matrix_b):
    return [[a * b for a, b in zip(matrix_a[0], matrix_b)]]

def transpose(matrix):
    return [list(i) for i in zip(*matrix)]

def matrix_multiplication(matrix_A, matrix_B):
    return [[sum(a * b for a, b in zip(a_row, b_col)) for b_col in zip(*matrix_B)] for a_row in matrix_A]

def generate_row(size):
    matrix = [(1 / size) + np.random.rand() / 1000 for _ in range(size)]
    s = sum(matrix)
    return [m / s for m in matrix]


def next_move(fish, model):
    obs = transpose(model.B)
    alpha = dot_prod(model.pi, obs[fish[0]])

    for i in fish[1:]:
        alpha = matrix_multiplication(alpha, model.A)
        alpha = dot_prod(alpha, obs[i])

    return sum(alpha[0])


class HMM:
    def __init__(self, species, emissions) -> None:
        self.pi = [generate_row(species)]
        self.A = [generate_row(species) for _ in range(species)]
        self.B = [generate_row(emissions) for _ in range(species)]

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_pi(self, pi):
        self.pi = pi


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models = [HMM(1, N_EMISSIONS) for _ in range(N_SPECIES)]
        self.fish_obs = [(i,[]) for i in range(N_FISH)]

    def update_model(self, model_id):
        A, B, pi = baum_welch(self.models[model_id].A, self.models[model_id].B, self.models[model_id].pi, self.obs)
        self.models[model_id].set_A(A)
        self.models[model_id].set_B(B)
        self.models[model_id].set_pi(pi)

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        # return (step % N_FISH, random.randint(0, N_SPECIES - 1))

        for i in range(len(self.fish_obs)):
                self.fish_obs[i][1].append(observations[i])

        if step < 10:
            return None

        else:
            fish_id, obs = self.fish_obs.pop()
            fish_type = 0
            max = 0

            for model, idx in zip(self.models, range(N_SPECIES)):
                m = next_move(obs, model)

                if m > max:
                    max = m
                    fish_type = idx
            self.obs = obs
        
            return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        if not correct:
            self.update_model(true_type)
