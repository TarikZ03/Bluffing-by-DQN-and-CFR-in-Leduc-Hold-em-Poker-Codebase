from rlcard.games.base import Card


class LeducholdemDealer:

    def __init__(self, np_random):
        ''' Initialize a leducholdem dealer class
        '''
        self.np_random = np_random

        # CHANGE: Expanded from 4 ranks ['J', 'Q', 'K', 'A'] to full 13 ranks
        self.deck = [Card(suit, rank) for suit in ['S', 'H', 'D', 'C']
                     for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']]

        self.shuffle()
        self.pot = 0

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_card(self):
        """
        Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        """
        return self.deck.pop()
