#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import sys

epsilon = sys.float_info.epsilon

def forward(O, A, B, pi):

    forward_probs = [[0.0 for i in range(len(A[0]))] for j in range(O)]
    alpha_T = [0.0 for i in range(O)]
    pi = pi[0]

    for t in range(len(O)):
        alpha_t = 0.0
        alpha_ij = 0.0

        for i in range(len(A[0])):
            if t == 0:
                forward_probs[t][i] = pi[i] * B[i][O[t]]
                alpha_ij += pi[i] * B[i][O[t]]
                alpha_t = alpha_ij
            else:
                alpha_ij = 0.0
                for j in range(len(A[0])):
                    alpha_ij += forward_probs[t-1][j] * A[j][i] * B[i][O[t]]
                forward_probs[t][i] = alpha_ij
                alpha_t += alpha_ij
        
        alpha_T[t] = 1/(alpha_t + epsilon)
        forward_probs = [alpha_T[t] * forward_probs[t][k] for k in range(len(A[0]))]
    
    return forward_probs, alpha_T

def backward(O, A, B, alpha_T):
    pass



class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        pass

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

        return None

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
        pass
