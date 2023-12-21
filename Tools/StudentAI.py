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

    '''def get_move(self,move):
        if len(move) != 0: #This checks if the player is player 1 or 2. If player 1 then will initialize as such.
            self.board.make_move(move,self.opponent[self.color]) #This makes the enemy's move, the rest of the code decides on its own move.
        else:
            self.color = 1
        moves = self.board.get_all_possible_moves(self.color)
        #Move = a a lit of tuples [(1,1), (2,2)] basically moving a list from one to anothe .
        #Returns a 2d list of all the possible pieces that can be moved,and the possible moves that piece can make.
        index = randint(0,len(moves)-1)
        inner_index =  randint(0,len(moves[index])-1)
        move = moves[index][inner_index]
        self.board.make_move(move,self.color)
        return move'''

    def get_move(self, move):
        if len(move) != 0:  # This checks if the player is player 1 or 2. If player 1 then will initialize as such.
            self.board.make_move(move, self.opponent[
                self.color])  # This makes the enemy's move, the rest of the code decides on its own move.
        else:
            self.color = 1
        root = MonteCarloTreeSearchNode(state=copy.deepcopy(self.board), color=self.color)
        move = root.best_action()
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
        self.color_dict = {2: "W",
                           1: "B"}  # For some reason .is_win() uses characters as input instead of integers so here was the fix for it.
        # Man do I hate Python's flexible typing

    def win_loss_diff(self):
        wins = self._results[1]
        ties = self._results[0]
        losses = self._results[-1]  # I think focusing on losses is actually interfering with the scoring.
        return wins - losses + 0.75 * ties

    def visits(self):
        return self._number_of_visits

    def expand(self):
        # random_checker = random.randint(1, len(self._untried))-1
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
        # return self.state.

    def rollout(self, og_color):
        current_state = copy.deepcopy(self.state)  # state is the board
        color = self.color  # color is an integer

        counter = 0
        while current_state.is_win(self.color_dict[color]) == 0:  # Color dict connects the integer to the string
            if (counter > self.max_depth):
                break
            possible_moves = current_state.get_all_possible_moves(color)  # gets all the possible moves for said color.
            if len(possible_moves) == 0:
                return (color+1)%2, color
            # f = open("output.txt", "w")
            # f.write(str(len(possible_moves)))
            # f.write(str(self.state.get_all_possible_moves(color)))
            # f.close()

            move = self.rollout_rules(possible_moves)
            current_state.make_move(move, color)
            color = color % 2 + 1  # Change side
            counter += 1
        return current_state.is_win(self.color_dict[color]), color  # Change color back to first parent color.

    def backpropogate(self, result):
        self._number_of_visits += 1
        '''
        if(won_color == self.color):
            self
        self._results[result] += 1
        '''
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
            # f = open("output.txt", "w")
            # f.write(str(c._results[1])+'\n' + str(c._results[-1]) + '\n')

            weights.append(c.win_loss_diff() / c.visits() + c_param * math.sqrt(math.log(self.visits()) / c.visits()))
            # weights.append(c.win_loss_diff()/ c.visits() + c_param *math.sqrt(math.log(self.visits())/ c.visits()))
        return self.children[max(enumerate(weights), key=lambda x: x[1])[0]]

    def rollout_rules(self, possiblemoves):

        # f = open("output.txt", "w")
        # f.write(str(len(possiblemoves)))
        # f.close()

        random1 = possiblemoves[random.randrange(len(possiblemoves))]
        random2 = random1[random.randrange(len(random1))]
        return random2

    def _tree_policy(self):
        current = self
        # might need to add depth here?
        counter = 0
        while not current.is_terminal_node():
            if (counter > self.max_depth):
                break
            if not current.is_fully_expanded():
                return current.expand()
            if(random.randint(1,15) % 6 == 1):
                current = random.choice(current.children)
            else:
                current = current.best_child()
            counter += 1
        return current

    def best_action(self):
        sims = 200
        for i in range(sims):
            node = self._tree_policy()
            roll_num, color = node.rollout(
                node.color)  # 1, 2 for black and white, and 0 or -1 for ties / inconclusiveness
            #reward = (1 if roll_num == color else 0 if (roll_num == 0 or roll_num == -1) else -1)
            # reward = (1 if roll_num == node.color else 0 if (roll_num == 0 or roll_num == -1) else -1)
            # won_color = roll_num
            node.backpropogate(roll_num)
            # node.backpropogate(won_color)
        bestmove = self.best_child().parent_action
        if bestmove:
            return bestmove
        else:
            return self.state.get_all_possible_moves(self.color)[0][0]

        return bestmove


'''
def Monte_Carlo_Search(state):
    tree <- Node(state)
    while time_remaining():
        leaf <- selectr(tree)
        child <-expand(leaf)
        result <-simulate(child)
        back_prop(result, child)
    return move with highest number.
'''
'''node = self._tree_policy()
roll_num = node.rollout()
reward = (1 if roll_num == self.color else 0 if roll_num == 0 else -1)
node.backpropogate(reward)
'''
