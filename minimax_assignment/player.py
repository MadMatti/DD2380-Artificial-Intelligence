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
        self.transposition_table = {}

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
        children_nodes.sort(key=self.heuristic_score, reverse=True)


        node_scores = [self.alphabeta_pruning(child, depth, alpha, beta, 1, initial_time) for child in children_nodes]
        best_score_index = node_scores.index(max(node_scores))

        return children_nodes[best_score_index].move

    def alphabeta_pruning(self, game_node, depth_level, alpha, beta, curr_player, start_time):
        
        state_hash = self.get_hash_state(game_node)
        if state_hash in self.transposition_table:
            depth, score = self.transposition_table[state_hash]
            if depth >= depth_level:
                return score

        if time.time() - start_time > 0.055:
            raise TimeoutError

        node_children = game_node.compute_and_get_children()
        node_children.sort(key=self.heuristic_score, reverse=True)

        if depth_level == 0 or len(node_children) == 0:
            heuristic_value = self.heuristic_score(game_node)
            self.transposition_table[state_hash] = (depth_level, heuristic_value)
            return heuristic_value

        elif curr_player == 0:
            heuristic_value = -math.inf
            for child in node_children:
                heuristic_value = max(heuristic_value,
                                      self.alphabeta_pruning(child, depth_level - 1, alpha, beta, 1, start_time))
                alpha = max(alpha, heuristic_value)
                if alpha >= beta:
                    break

        else:
            heuristic_value = math.inf
            for child in node_children:
                heuristic_value = min(heuristic_value,
                                      self.alphabeta_pruning(child, depth_level - 1, alpha, beta, 0, start_time))
                beta = min(beta, heuristic_value)
                if alpha >= beta:
                    break
        
        self.transposition_table[state_hash] = (depth_level, heuristic_value)

        return heuristic_value

    def heuristic_score(self, game_node):
        """
        Heuristic score for the minimax algorithm
        :param node: Current game tree node
        :type node: game_tree.Node
        :return: heuristic score
        :rtype: float
        """
        player_score = game_node.state.player_scores[0]
        opponent_score = game_node.state.player_scores[1]
        score_difference = player_score - opponent_score
    
        heuristic_value = 0
    
        # Factor in distance to nearest fish and fish scores
        for fish_id, fish_pos in game_node.state.fish_positions.items():
            fish_score = game_node.state.fish_scores[fish_id]
            player_distance = self.manhattan_distance(fish_pos, game_node.state.hook_positions[0])
            opponent_distance = self.manhattan_distance(fish_pos, game_node.state.hook_positions[1])
    
            # Fishes that are closer and with higher score are more valuable
            if player_distance != 0:
                heuristic_value += (fish_score / player_distance)
            if opponent_distance != 0:
                heuristic_value -= (fish_score / opponent_distance) * 0.5
    
        # If the player got at least one fish, add a bonus to the heuristic value
        if game_node.state.player_caught[0] != -1:
            caught_fish_score = game_node.state.fish_scores[game_node.state.player_caught[0]]
            heuristic_value += caught_fish_score * 2 
    
        heuristic_value += 2.5 * score_difference
        heuristic_value += 0.1 * game_node.depth
    
        return heuristic_value

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


    def get_hash_state(self, game_node):
        """
        Generate a unique hash for a given game state.
        :param game_node: The current game node
        :return: A unique hash string
        """
        state = game_node.state
        fish_positions = state.fish_positions
        hook_positions = state.hook_positions
        hash_key = str(hook_positions) + str(fish_positions)
        return hash_key