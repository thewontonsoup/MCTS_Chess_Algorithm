from random import randint
from BoardClasses import Move
from BoardClasses import Board
import copy
from collections import defaultdict
import math
from itertools import chain
import time
import numpy as np
# The following part should be completed by students.
# Students can modify anything except the class name and exisiting functions and varibles.
import random


class StudentAI():

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1: 2, 2: 1}
        self.color = 2
        self.root = None
        self.start_time = time.time()

    def get_move(self, move):
        if len(move) != 0:  # This checks if the player is player 1 or 2. If player 1 then will initialize as such.
            self.board.make_move(move, self.opponent[self.color])
            # This makes the enemy's move, the rest of the code decides on its own move.
        else:
            self.color = 1
        if (self.root == None):
            self.root = MonteCarloTreeSearchNode(state=copy.deepcopy(self.board), color=self.color)
        else:
            self.root = self.root.update_tree(move, self.opponent[self.color])
        self.root, move = self.root.best_action(self.start_time)
        self.board.make_move(move, self.color)
        return move


class MonteCarloTreeSearchNode():
    def __init__(self, state, color, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0  # wins is key = +1 and losses is key = -1
        self._results[0] = 0  # neutral state for ties and not game overs I guess
        self._results[-1] = 0
        self.color = color

        # in cases of infinite loops
        average_branching_factor = 6
        average_depth = 18
        dimension = len(self.state.board[0]) * len(self.state.board)
        self.max_depth = average_branching_factor * average_depth * dimension

        self._untried = list(chain.from_iterable(self.state.get_all_possible_moves(self.color)))  # converts 2d to 1D
        self.color_dict = {2: "W", 1: "B"}

    def update_tree(self, move, opp_color):
        for child in self.children:
            if (child.parent_action and child.parent_action.seq == move.seq):
                return child
        self.state.make_move(move, opp_color)
        child_node = MonteCarloTreeSearchNode(state=copy.deepcopy(self.state), color=(opp_color % 2) + 1, parent=None,
                                              parent_action=move)
        return child_node

    def win_loss_diff(self):
        wins = self._results[1]
        ties = self._results[0]
        losses = self._results[-1]  # I think focusing on losses is actually interfering with the scoring.
        return wins - losses + ties

    def visits(self):
        return self._number_of_visits

    def expand(self):
        move = self._untried.pop()
        self.state.make_move(move, self.color)
        child_node = MonteCarloTreeSearchNode(state=copy.deepcopy(self.state), color=(self.color % 2) + 1, parent=self,
                                              parent_action=move)
        self.state.undo()
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return (len(self.children) == 0 and len(self._untried) == 0) or self.state.is_win(
            self.color_dict[self.color]) != 0

    def rollout(self, og_color):
        current_state = copy.deepcopy(self.state)  # state is the board
        color = self.color  # color is an integer

        counter = 0
        while current_state.is_win(self.color_dict[color]) == 0:  # Color dict connects the integer to the string
            if (counter > self.max_depth):
                break
            possible_moves = current_state.get_all_possible_moves(color)  # gets all the possible moves for said color.
            if len(possible_moves) == 0:
                return (color + 1) % 2, color
            move = self.rollout_rules(possible_moves)
            current_state.make_move(move, color)
            color = color % 2 + 1  # Change side
            counter += 1
        return current_state.is_win(self.color_dict[color]), color  # Change color back to first parent color.

    def backpropogate(self, result):
        self._number_of_visits += 1
        reward = (1 if result == self.color else 0 if (result == 0 or result == -1) else -1)
        self._results[reward] += 1
        if self.parent:  # if it is not the root.
            self.parent.backpropogate(result)

    def is_fully_expanded(self):
        return len(self._untried) == 0

    def best_child(self, c_param=math.sqrt(2)):  # This parameter could change possibly if there are better outcomes.
        weights = []
        if len(self.children) == 0:
            return self
        for c in self.children:
            weights.append(c.win_loss_diff() / c.visits() + c_param * math.sqrt(math.log(self.visits()) / c.visits()))
        return self.children[max(enumerate(weights), key=lambda x: x[1])[0]]

    def rollout_rules(self, possiblemoves):
        random1 = possiblemoves[random.randrange(len(possiblemoves))]
        random2 = random1[random.randrange(len(random1))]
        return random2

    def _tree_policy(self):
        current = self
        for _ in range(self.max_depth):
            if current.is_terminal_node():
                break
            if not current.is_fully_expanded():
                return current.expand()
            if random.randint(1, 15) % 6 == 1:
                current = random.choice(current.children)
            else:
                current = current.best_child()
        return current

    def best_action(self, start_time):
        cur_time = time.time()
        sims = 185
        duration = (4 if (cur_time - start_time) < 20 else 2 if (cur_time - start_time >= 240) else -1)
        for i in range(sims):
            if (duration > 0 and time.time() - cur_time >= duration):
                break
            node = self._tree_policy()
            roll_num, color = node.rollout(
                node.color)  # 1, 2 for black and white, and 0 or -1 for ties / inconclusiveness
            node.backpropogate(roll_num)
        bestchild = self.best_child()
        bestmove = bestchild.parent_action
        if bestmove:
            return bestchild, bestmove
        else:
            return bestchild, self.state.get_all_possible_moves(self.color)[0][0]
