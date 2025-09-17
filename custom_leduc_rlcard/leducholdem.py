import json
import os
import numpy as np
from collections import OrderedDict
import rlcard
from rlcard.envs import Env
from BNAIC_paper_files.custom_leduc_rlcard import Game
from rlcard.utils import *

DEFAULT_GAME_CONFIG = {
    'game_num_players': 2,
}


class LeducholdemEnv(Env):
    ''' Leduc Hold'em Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'custom-leduc-holdem'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']

        # CHANGE: Updated from [36] to [156] to accommodate 52 cards
        # Original: 16 (hand) + 16 (public) + 4 (chips) = 36
        # New: 52 (hand) + 52 (public) + 52 (chips) = 156
        self.state_shape = [[156] for _ in range(self.num_players)]

        self.action_shape = [None for _ in range(self.num_players)]

        card_path = os.path.join(os.path.dirname(__file__), 'card2index.json')
        with open(card_path, 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all leagal actions - UNCHANGED

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent - UPDATED for 52 cards

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        public_card = state['public_card']
        hand = state['hand']

        # CHANGE: Expanded observation vector from 36 to 156
        obs = np.zeros(156)

        # CHANGE: Hand encoding now uses 52 positions instead of 16
        obs[self.card2index[hand]] = 1

        # CHANGE: Public card encoding uses positions 52-103 instead of 16-19
        if public_card:
            obs[self.card2index[public_card] + 52] = 1  # Offset by 52

        # CHANGE: Chip encoding uses positions 104-155 instead of 22-35
        # Keep same logic but with more positions available
        my_chips = min(state['my_chips'], 25)  # Cap for encoding
        total_opponent_chips = min(sum(state['all_chips']) - state['my_chips'], 25)

        obs[104 + my_chips] = 1  # My chips (positions 104-129)
        obs[130 + total_opponent_chips] = 1  # Opponent chips (positions 130-155)

        extracted_state['obs'] = obs
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game - UNCHANGED

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game - UNCHANGED

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        ''' Get the perfect information of the current state - UNCHANGED

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = self.game.public_card.get_index() if self.game.public_card else None
        state['hand_cards'] = [self.game.players[i].hand.get_index() for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state
