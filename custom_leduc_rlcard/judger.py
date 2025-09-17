# Replace your current judger.py with this simplified version:

from rlcard.utils.utils import rank2int
from rlcard.games.base import Card


class LeducholdemJudger:
    ''' Simplified Judger class for Leduc Hold'em - NO STRAIGHTS '''

    def __init__(self, np_random):
        ''' Initialize a judger class '''
        self.np_random = np_random

        # Full 13 ranks
        self.RANK_ORDER = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                           'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

        # Suit ordering - HIGHER SUIT WINS
        self.SUIT_ORDER = {'C': 0, 'D': 1, 'H': 2, 'S': 3}  # lowest to highest

    def hand_score(self, card):
        """Assign a score based on rank and suit"""
        return self.RANK_ORDER[card.rank] * 4 + self.SUIT_ORDER[card.suit]

    def judge_game(self, players, public_card):
        ''' Simplified judging: ONLY pairs and high cards - NO STRAIGHTS

        Hand ranking (highest to lowest):
        1. Pair (matching rank with public card) - tiebreak by suit
        2. High card - tiebreak by suit automatically via hand_score
        '''
        num_players = len(players)
        winners = [0 for _ in range(num_players)]
        fold_count = sum([1 for p in players if p.status == 'folded'])

        # If only one player remains (all others folded)
        if fold_count == num_players - 1:
            for i, p in enumerate(players):
                if p.status != 'folded':
                    winners[i] = 1
            return self.calculate_payoffs(winners, players)

        # If no public card yet (game ended in first round), compare hole cards
        if public_card is None:
            scores = [(i, self.hand_score(p.hand)) for i, p in enumerate(players) if p.status != 'folded']
            max_score = max(score for _, score in scores)
            for i, score in scores:
                if score == max_score:
                    winners[i] = 1
            return self.calculate_payoffs(winners, players)

        # Validate cards
        assert public_card.rank in self.RANK_ORDER, f"Unrecognized public rank: {public_card.rank}"
        assert public_card.suit in self.SUIT_ORDER, f"Unrecognized public suit: {public_card.suit}"
        for p in players:
            if p.status != 'folded':
                assert p.hand.rank in self.RANK_ORDER, f"Unrecognized player rank: {p.hand.rank}"
                assert p.hand.suit in self.SUIT_ORDER, f"Unrecognized player suit: {p.hand.suit}"

        # Check for pairs (matching rank with public card)
        pair_players = []
        non_pair_players = []

        for i, p in enumerate(players):
            if p.status != 'folded':
                if p.hand.rank == public_card.rank:
                    # Player has a pair - use hand score for tiebreaking
                    pair_score = self.hand_score(p.hand)
                    pair_players.append((i, pair_score))
                else:
                    # Player has high card only
                    high_card_score = self.hand_score(p.hand)
                    non_pair_players.append((i, high_card_score))

        # Pairs always beat high cards
        if pair_players:
            # Find the best pair (highest suit wins among pairs)
            max_pair_score = max(score for _, score in pair_players)
            for i, score in pair_players:
                if score == max_pair_score:
                    winners[i] = 1
        else:
            # No pairs, compare high cards
            max_score = max(score for _, score in non_pair_players)
            for i, score in non_pair_players:
                if score == max_score:
                    winners[i] = 1

        return self.calculate_payoffs(winners, players)

    def calculate_payoffs(self, winners, players):
        total_chips = sum([p.in_chips for p in players])
        win_share = float(total_chips) / sum(winners)
        payoffs = []
        for i, p in enumerate(players):
            payoff = win_share - p.in_chips if winners[i] == 1 else -p.in_chips
            payoffs.append(payoff)
        return payoffs
