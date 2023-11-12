#!/usr/bin/env python3
import random
import math
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        initial_time = time.time()
        current_depth = 0
        timeout = False
        best_move = 0

        while not timeout:
            try:
                curr_move = self.best_move_search(initial_tree_node, current_depth, initial_time)
                current_depth += 1
                best_move = curr_move
            except:
                timeout = True
        
        return ACTION_TO_STR[best_move]


    def best_move_search(self, root_node, depth, initial_time):
        """
        Search for the best move using minimax
        :param node: Initial game tree node
        :type node: game_tree.Node
        :param depth: Current depth of the search
        :type depth: int
        :param initial_time: Initial time of the search
        :type initial_time: float
        :param visited_nodes: Dictionary of visited nodes
        :type visited_nodes: dict
        :return: best move
        :rtype: int
        """
        alpha = -math.inf
        beta = math.inf

        children_nodes = root_node.compute_and_get_children()

        node_scores = [self.alphabeta_pruning(child, depth, alpha, beta, 1, initial_time) for child in children_nodes]
        best_score_index = node_scores.index(max(node_scores))

        return children_nodes[best_score_index].move


    def alphabeta_pruning(self, game_node, depth_level, alpha, beta, curr_player, start_time):

        if time.time() - start_time > 0.055:
            raise TimeoutError

        node_children = game_node.compute_and_get_children()
        node_children.sort(key=self.heuristic_score, reverse=True)

        if depth_level == 0 or len(node_children) == 0:
            heuristic_value = self.heuristic_score(game_node)

        elif curr_player == 0:
            heuristic_value = -math.inf
            for child in node_children:
                heuristic_value = max(heuristic_value, self.alphabeta_pruning(child, depth_level - 1, alpha, beta, 1, start_time))
                alpha = max(alpha, heuristic_value)
                if alpha >= beta:
                    break

        else:
            heuristic_value = math.inf
            for child in node_children:
                heuristic_value = min(heuristic_value, self.alphabeta_pruning(child, depth_level - 1, alpha, beta, 0, start_time))
                beta = min(beta, heuristic_value)
                if alpha >= beta:
                    break

        return heuristic_value


    def heuristic_score(self, game_node):
        """
        Heuristic score for the minimax algorithm
        :param node: Current game tree node
        :type node: game_tree.Node
        :return: heuristic score
        :rtype: float
        """
        score_difference = game_node.state.player_scores[0] - game_node.state.player_scores[1]
        heuristic_value = 0

        for fish_id in game_node.state.fish_positions:
            distance = self.manhattan_distance(game_node.state.fish_positions[fish_id], game_node.state.hook_positions[0])

            if distance == 0 and game_node.state.fish_scores[fish_id] > 0:
                return math.inf
            heuristic_value = max(heuristic_value, game_node.state.fish_scores[fish_id] / math.exp(distance))

        return heuristic_value + 2*score_difference

    
    def manhattan_distance(self, pos1, pos2):
        """
        Manhattan distance between two positions
        :param pos1: First position
        :type pos1: tuple
        :param pos2: Second position
        :type pos2: tuple
        :return: Manhattan distance
        :rtype: int
        """
        y_dist = abs(pos1[1] - pos2[1])
        x_delta = abs(pos1[0] - pos2[0])
        x_dist = min(x_delta, 20 - x_delta)
        return x_dist + y_dist
        
        
        

        
