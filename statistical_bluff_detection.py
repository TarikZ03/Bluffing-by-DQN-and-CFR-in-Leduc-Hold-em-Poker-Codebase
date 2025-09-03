import numpy as np
from collections import defaultdict, Counter
import json
import pickle
import matplotlib.pyplot as plt
import wandb
import os
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class StatisticalBluffDetector52:
    """
    Statistical bluff detection for 52-card Custom Leduc Hold'em
    """

    def __init__(self, rank_order={'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                                   'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12},
                 suit_order={'C': 0, 'D': 1, 'H': 2, 'S': 3}):
        self.rank_order = rank_order
        self.suit_order = suit_order

        # Store historical action frequencies for belief distribution
        self.action_history = defaultdict(lambda: defaultdict(lambda: {'pairs': [], 'non_pairs': []}))

        # Store expected utilities for EV calculation
        self.ev_history = defaultdict(lambda: defaultdict(list))

    def card_score(self, card_str):
        """Calculate deterministic card score for 52-card deck"""
        if len(card_str) >= 2:
            suit = card_str[0]
            rank = card_str[1]
            return self.rank_order.get(rank, 0) * 4 + self.suit_order.get(suit, 0)
        return 0

    def hand_strength(self, hand, public_card=None):
        """
        Calculate hand strength s(h).
        Returns: (base_score, is_pair)
        """
        base_score = self.card_score(hand)
        is_pair = False

        if (public_card
                and len(public_card) >= 2):
            if hand[1] == public_card[1]:  # Pair
                is_pair = True

        return base_score, is_pair

    def get_context(self, public_card=None, betting_round=0, position=0):
        """Create a context identifier pc."""
        pc_str = "none" if public_card is None else public_card
        return f"pc:{pc_str}_round:{betting_round}_pos:{position}"

    def update_belief_distribution(self, context, action, hand_strength, is_pair):
        """Update our belief distribution based on observed actions"""
        if is_pair:
            self.action_history[context][action]['pairs'].append(hand_strength)
        else:
            self.action_history[context][action]['non_pairs'].append(hand_strength)

    def get_belief_distribution(self, context, action):
        """
        Get belief distribution μ(h'|a,pc) for hands that typically take action a in context pc.
        Returns separate statistics for pairs and non-pairs.
        """
        data = self.action_history[context][action]

        pairs = data['pairs']
        non_pairs = data['non_pairs']

        # Calculate pair frequency
        total = len(pairs) + len(non_pairs)
        if total < 5:
            return None

        pair_freq = len(pairs) / total

        # Statistics for non-pairs
        non_pair_mean = np.mean(non_pairs) if non_pairs else 0
        non_pair_std = np.std(non_pairs) if len(non_pairs) > 1 else 1

        # Statistics for pairs (if any)
        pair_mean = np.mean(pairs) if pairs else 0
        pair_std = np.std(pairs) if len(pairs) > 1 else 1

        return {
            'total_samples': total,
            'pair_frequency': pair_freq,
            'non_pair_mean': non_pair_mean,
            'non_pair_std': non_pair_std,
            'pair_mean': pair_mean,
            'pair_std': pair_std,
            'has_pairs': len(pairs) > 0
        }

    def update_ev(self, context, hand, action, payoff):
        """Update expected value history"""
        key = (hand, action)
        self.ev_history[context][key].append(payoff)

    def get_expected_utility(self, context, hand, action):
        """Get expected utility u(h,a) based on historical data"""
        key = (hand, action)
        payoffs = self.ev_history[context].get(key, [])

        if len(payoffs) < 3:
            return None

        return np.mean(payoffs)

    def is_statistical_bluff(self, hand, action, context, passive_action='call', std_threshold=0.5):
        """
        Determine if an action is a bluff based on the statistical definition.
        """
        # Only raises can be bluffs in our simplified model
        if action != 'raise':
            return False, {}

        # Calculate hand strength
        public_card = None
        if 'pc:' in context:
            pc_part = context.split('_')[0].replace('pc:', '')
            if pc_part != 'none':
                public_card = pc_part

        h_strength, h_is_pair = self.hand_strength(hand, public_card)

        # Get belief distribution for this action in this context
        belief_stats = self.get_belief_distribution(context, action)

        if belief_stats is None:
            return False, {"reason": "insufficient_belief_data"}

        # Determine if hand is weaker than typical
        misrepresents = False

        if h_is_pair:
            # For pairs, compare against other pairs
            if belief_stats['has_pairs']:
                # Compare against pair distribution
                threshold = belief_stats['pair_mean'] - std_threshold * belief_stats['pair_std']
                misrepresents = h_strength < threshold
            else:
                # If pairs rarely raise in this context, this pair raise is not a bluff
                misrepresents = False
        else:
            # For non-pairs, check two things:
            # 1. Is raising without a pair unusual? (high pair frequency means yes)
            # 2. Is this non-pair weaker than typical non-pairs that raise?

            # If most raises are pairs, then non-pair raises are suspicious
            if belief_stats['pair_frequency'] > 0.7:  # 70% or more are pairs
                misrepresents = True
            else:
                # Compare against non-pair distribution
                threshold = belief_stats['non_pair_mean'] - std_threshold * belief_stats['non_pair_std']
                misrepresents = h_strength < threshold

        # Get expected utilities
        ev_aggressive = self.get_expected_utility(context, hand, action)
        ev_passive = self.get_expected_utility(context, hand, passive_action)

        if ev_aggressive is None or ev_passive is None:
            # Can still be a bluff based on misrepresentation alone
            if misrepresents:
                return True, {
                    "reason": "misrepresentation_only",
                    "hand_strength": h_strength,
                    "is_pair": h_is_pair,
                    "belief_stats": belief_stats
                }
            return False, {"reason": "insufficient_ev_data"}

        # Condition 2: EV check
        ev_positive = ev_aggressive > ev_passive

        # Both conditions must be true for a statistical bluff
        is_bluff = misrepresents and ev_positive

        details = {
            "hand_strength": h_strength,
            "is_pair": h_is_pair,
            "belief_stats": belief_stats,
            "misrepresents": misrepresents,
            "ev_aggressive": ev_aggressive,
            "ev_passive": ev_passive,
            "ev_positive": ev_positive,
            "is_bluff": is_bluff
        }

        if is_bluff:
            details["reason"] = "both_conditions_met"
        else:
            details["reason"] = "not_a_bluff"

        return is_bluff, details


def create_belief_distribution_visualization(detector, player_name):
    """Create simple belief distribution visualization"""

    # Find the most active context
    best_context = None
    max_raises = 0

    for context, actions in detector.action_history.items():
        if 'raise' in actions:
            raise_data = actions['raise']
            total = len(raise_data['pairs']) + len(raise_data['non_pairs'])
            if total > max_raises:
                max_raises = total
                best_context = context

    if not best_context or max_raises < 20:
        return

    # Get the raise data
    raise_data = detector.action_history[best_context]['raise']
    pair_count = len(raise_data['pairs'])
    non_pair_count = len(raise_data['non_pairs'])

    print(f"Selected context: {best_context}")
    print(f"Pairs: {pair_count}, Non-pairs: {non_pair_count}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    ax1.pie([pair_count, non_pair_count], labels=['Pairs', 'Non-pairs'], autopct='%1.1f%%',
            colors=['gold', 'lightblue'])
    ax1.set_title(f'{player_name}: Pair vs Non-pair Raises')

    # Histogram
    if raise_data['non_pairs']:
        ax2.hist(raise_data['non_pairs'], bins=range(0, 53, 2), alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_xlabel('Hand Strength')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{player_name}: Non-pair Strength Distribution')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({f'{player_name} Belief Distribution': wandb.Image(plt)})
    plt.close()

def analyze_statistical_bluffs_52card(log_path, player_id=0, player_name="DQN"):
    """
    Analyze statistical bluffs for 52-card version with comparable output to threshold analysis
    """
    detector = StatisticalBluffDetector52()

    print(f"=== Building Statistical Model for {player_name} ===")

    # Initialize tracking similar to threshold analysis
    total_games = 0
    player_total_actions = 0
    total_statistical_bluff_attempts = 0
    total_statistical_bluff_successes = 0

    # Detailed outcome tracking
    statistical_bluff_outcomes = {
        'opponent_folded': 0,
        'opponent_called': 0,
        'opponent_raised': 0,
        'opponent_checked': 0,
        'showdown_won': 0,
        'showdown_lost': 0
    }

    # Tracking by hand - ATTEMPTS
    statistical_bluff_attempts_by_hand = Counter()
    statistical_bluff_attempts_by_rank = Counter()
    statistical_bluff_attempts_by_rank_group = Counter()

    # Tracking by hand - SUCCESSES
    statistical_bluff_successes_by_hand = Counter()
    statistical_bluff_successes_by_rank = Counter()
    statistical_bluff_successes_by_rank_group = Counter()

    # Opponent reactions
    opponent_reactions_to_statistical_bluffs = Counter()
    statistical_reactions_before_public = Counter()
    statistical_reactions_after_public = Counter()

    action_id_to_name = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}

    def get_rank_group(hand):
        if len(hand) >= 2:
            rank = hand[1]
            if rank in ['2', '3', '4', '5', '6']:
                return 'Low (2-6)'
            elif rank in ['7', '8', '9', 'T']:
                return 'Medium (7-T)'
            elif rank in ['J', 'Q']:
                return 'High (J-Q)'
            elif rank in ['K', 'A']:
                return 'Premium (K-A)'
        return 'Unknown'

    def get_hand_category(hand):
        if len(hand) >= 2:
            rank = hand[1]
            suit = hand[0]
            return f"{rank}{suit}"
        return hand

    # First pass: Build belief distributions and EV data
    print("Pass 1: Building belief distributions...")
    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            log = data['log']
            payoffs = data['payoffs']
            total_games += 1

            # Track game state
            game_states = {}
            betting_round = 0

            for i, entry in enumerate(log):
                pid = entry['player_id']
                hand = entry.get('hand', '')
                action = entry.get('action_taken', -1)
                public_card = entry.get('public_card')

                # Update betting round
                if public_card and betting_round == 0:
                    betting_round = 1

                # Map action index to name
                action_name = action_id_to_name.get(action, 'unknown')

                if hand and action_name != 'unknown':
                    # Create context
                    position = pid
                    context = detector.get_context(public_card, betting_round, position)

                    # Calculate hand strength
                    strength, is_pair = detector.hand_strength(hand, public_card)

                    # Update belief distribution
                    detector.update_belief_distribution(context, action_name, strength, is_pair)

                    # Store for EV calculation
                    if pid not in game_states:
                        game_states[pid] = {'hand': hand, 'actions': []}
                    game_states[pid]['actions'].append((context, action_name))

            # Update EVs based on final payoffs
            for pid, state in game_states.items():
                for context, action in state['actions']:
                    detector.update_ev(context, state['hand'], action, payoffs[pid])

    # Second pass: Detect statistical bluffs
    print("Pass 2: Detecting statistical bluffs...")
    with open(log_path, 'r') as f:
        for game_num, line in enumerate(f):
            data = json.loads(line)
            log = data['log']
            payoffs = data['payoffs']
            betting_round = 0

            for i, entry in enumerate(log):
                pid = entry['player_id']

                if pid == player_id:
                    player_total_actions += 1

                    hand = entry.get('hand', '')
                    action = entry.get('action_taken', -1)
                    public_card = entry.get('public_card')

                    if public_card and betting_round == 0:
                        betting_round = 1

                    action_name = action_id_to_name.get(action, 'unknown')

                    if action_name == 'raise' and hand:
                        # Check if it's a statistical bluff
                        context = detector.get_context(public_card, betting_round, pid)
                        is_bluff, details = detector.is_statistical_bluff(hand, action_name, context)

                        if is_bluff:
                            total_statistical_bluff_attempts += 1

                            # Track attempt details
                            hand_cat = get_hand_category(hand)
                            rank_group = get_rank_group(hand)

                            statistical_bluff_attempts_by_hand[hand_cat] += 1
                            statistical_bluff_attempts_by_rank[hand[1]] += 1
                            statistical_bluff_attempts_by_rank_group[rank_group] += 1

                            # Look for opponent's reaction
                            opponent_reaction = None
                            opponent_action_index = None
                            reaction_context = None

                            for j in range(i + 1, len(log)):
                                next_entry = log[j]
                                if next_entry['player_id'] != player_id:  # Opponent's turn
                                    opponent_action_index = next_entry.get('action_taken', -1)
                                    opponent_reaction = action_id_to_name.get(opponent_action_index, 'UNKNOWN')
                                    reaction_context = next_entry.get('public_card', None)
                                    break

                            # Track reaction and determine success
                            if opponent_reaction and opponent_reaction != 'UNKNOWN':
                                opponent_reactions_to_statistical_bluffs[opponent_reaction] += 1

                                # Context tracking
                                if reaction_context:
                                    statistical_reactions_after_public[opponent_reaction] += 1
                                else:
                                    statistical_reactions_before_public[opponent_reaction] += 1

                                # Determine outcome
                                if opponent_reaction == 'fold':
                                    # STATISTICAL BLUFF SUCCESS!
                                    total_statistical_bluff_successes += 1
                                    statistical_bluff_outcomes['opponent_folded'] += 1

                                    # Track successful bluff by hand
                                    statistical_bluff_successes_by_hand[hand_cat] += 1
                                    statistical_bluff_successes_by_rank[hand[1]] += 1
                                    statistical_bluff_successes_by_rank_group[rank_group] += 1

                                elif opponent_reaction == 'call':
                                    statistical_bluff_outcomes['opponent_called'] += 1
                                    # Check final showdown result
                                    if payoffs[player_id] > 0:
                                        statistical_bluff_outcomes['showdown_won'] += 1
                                    else:
                                        statistical_bluff_outcomes['showdown_lost'] += 1

                                elif opponent_reaction == 'raise':
                                    statistical_bluff_outcomes['opponent_raised'] += 1
                                elif opponent_reaction == 'check':
                                    statistical_bluff_outcomes['opponent_checked'] += 1

    # Calculate metrics
    statistical_bluff_attempt_rate = total_statistical_bluff_attempts / player_total_actions if player_total_actions > 0 else 0
    statistical_bluff_success_rate = total_statistical_bluff_successes / total_statistical_bluff_attempts if total_statistical_bluff_attempts > 0 else 0

    immediate_successes = statistical_bluff_outcomes['opponent_folded']
    lucky_wins = statistical_bluff_outcomes['showdown_won']
    total_positive_outcomes = immediate_successes + lucky_wins

    # Print results comparable to threshold analysis
    print("\n" + "=" * 80)
    print(f"{player_name} STATISTICAL BLUFF ANALYSIS - ATTEMPTS vs SUCCESSES")
    print("=" * 80)

    print(f"\n=== BASIC METRICS ===")
    print(f"Total Games: {total_games}")
    print(f"{player_name} Total Actions: {player_total_actions}")
    print(f"")
    print(f"STATISTICAL BLUFF ATTEMPTS: {total_statistical_bluff_attempts}")
    print(
        f"Statistical Bluff Attempt Rate: {statistical_bluff_attempt_rate:.3f} ({statistical_bluff_attempt_rate * 100:.1f}%)")
    print(f"")
    print(f"STATISTICAL BLUFF SUCCESSES (opponent folded): {total_statistical_bluff_successes}")
    print(
        f"Statistical Bluff Success Rate: {statistical_bluff_success_rate:.3f} ({statistical_bluff_success_rate * 100:.1f}%)")

    print(f"\n=== OPPONENT REACTIONS TO STATISTICAL BLUFF ATTEMPTS ===")
    total_reactions = sum(opponent_reactions_to_statistical_bluffs.values())
    for reaction, count in opponent_reactions_to_statistical_bluffs.most_common():
        percentage = (count / total_reactions) * 100 if total_reactions > 0 else 0
        print(f"  {reaction}: {count} ({percentage:.1f}%)")

    print(f"\n=== DETAILED STATISTICAL BLUFF ATTEMPT OUTCOMES ===")
    for outcome, count in statistical_bluff_outcomes.items():
        percentage = (count / total_statistical_bluff_attempts) * 100 if total_statistical_bluff_attempts > 0 else 0
        print(f"  {outcome.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    print(f"\n=== SUCCESS BREAKDOWN ===")
    print(
        f"  Immediate successes (opponent folded): {immediate_successes}/{total_statistical_bluff_attempts} ({immediate_successes / total_statistical_bluff_attempts * 100:.1f}%)")
    print(
        f"  Lucky wins (called but won): {lucky_wins}/{total_statistical_bluff_attempts} ({lucky_wins / total_statistical_bluff_attempts * 100:.1f}%)")
    print(
        f"  Total positive outcomes: {total_positive_outcomes}/{total_statistical_bluff_attempts} ({total_positive_outcomes / total_statistical_bluff_attempts * 100:.1f}%)")

    print(f"\n=== STATISTICAL BLUFF ATTEMPTS BY RANK GROUP ===")
    for group, count in statistical_bluff_attempts_by_rank_group.most_common():
        percentage = (count / total_statistical_bluff_attempts) * 100 if total_statistical_bluff_attempts > 0 else 0
        print(f"  {group}: {count} ({percentage:.1f}%)")

    print(f"\n=== STATISTICAL BLUFF SUCCESSES BY RANK GROUP ===")
    for group, count in statistical_bluff_successes_by_rank_group.most_common():
        success_rate = count / statistical_bluff_attempts_by_rank_group[group] * 100 if \
        statistical_bluff_attempts_by_rank_group[group] > 0 else 0
        print(
            f"  {group}: {count} successes / {statistical_bluff_attempts_by_rank_group[group]} attempts ({success_rate:.1f}%)")

    print(f"\n=== TOP 10 MOST ATTEMPTED STATISTICAL BLUFF HANDS ===")
    for hand, attempt_count in statistical_bluff_attempts_by_hand.most_common(10):
        success_count = statistical_bluff_successes_by_hand.get(hand, 0)
        success_rate = success_count / attempt_count * 100 if attempt_count > 0 else 0
        print(f"  {hand}: {attempt_count} attempts, {success_count} successes ({success_rate:.1f}%)")

    # Return data for visualization
    return {
        'detector': detector,
        'total_games': total_games,
        'player_total_actions': player_total_actions,
        'total_statistical_bluff_attempts': total_statistical_bluff_attempts,
        'total_statistical_bluff_successes': total_statistical_bluff_successes,
        'statistical_bluff_attempt_rate': statistical_bluff_attempt_rate,
        'statistical_bluff_success_rate': statistical_bluff_success_rate,
        'statistical_bluff_outcomes': statistical_bluff_outcomes,
        'statistical_bluff_attempts_by_hand': statistical_bluff_attempts_by_hand,
        'statistical_bluff_attempts_by_rank': statistical_bluff_attempts_by_rank,
        'statistical_bluff_attempts_by_rank_group': statistical_bluff_attempts_by_rank_group,
        'statistical_bluff_successes_by_hand': statistical_bluff_successes_by_hand,
        'statistical_bluff_successes_by_rank': statistical_bluff_successes_by_rank,
        'statistical_bluff_successes_by_rank_group': statistical_bluff_successes_by_rank_group,
        'opponent_reactions_to_statistical_bluffs': opponent_reactions_to_statistical_bluffs,
        'statistical_reactions_before_public': statistical_reactions_before_public,
        'statistical_reactions_after_public': statistical_reactions_after_public,
        'immediate_successes': immediate_successes,
        'lucky_wins': lucky_wins,
        'total_positive_outcomes': total_positive_outcomes
    }


def create_comparable_visualizations(data, player_name, project_name='BNAIC-statistical-bluff-analysis-52card'):
    """Create visualizations comparable to threshold analysis"""

    wandb.init(project=project_name, name=f'{player_name}_Statistical_Bluff_Analysis_52Card')

    create_belief_distribution_visualization(data['detector'], player_name)

    def plot_bar(data_dict, title, xlabel, ylabel, color='red', figsize=(10, 6)):
        if not data_dict:
            print(f"No data to plot for: {title}")
            return

        sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:15]  # Top 15
        keys, values = zip(*sorted_items) if sorted_items else ([], [])

        plt.figure(figsize=figsize)
        bars = plt.bar(keys, values, color=color, alpha=0.7, edgecolor='black')

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value}', ha='center', va='bottom', fontsize=8)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        wandb.log({title: wandb.Image(plt)})
        plt.close()

    # Determine opponent name and color for reaction charts
    opponent_name = "CFR" if player_name == "DQN" else "DQN"
    reaction_color = "red" if opponent_name == "CFR" else "blue"

    # 1. Opponent reactions to statistical bluff attempts
    if data['opponent_reactions_to_statistical_bluffs']:
        plot_bar(data['opponent_reactions_to_statistical_bluffs'],
                 f'{opponent_name} Reactions to {player_name} Bluff Attempts (Statistics-based)',
                 f'{opponent_name} Action', 'Count', color=reaction_color)

    # 2. Pre/Post flop context
    if data['statistical_reactions_before_public']:
        plot_bar(data['statistical_reactions_before_public'],
                 f'{opponent_name} Reactions to {player_name} Bluff Attempts - Pre-flop (Statistics-based)',
                 f'{opponent_name} Action', 'Count', color=reaction_color)
    if data['statistical_reactions_after_public']:
        plot_bar(data['statistical_reactions_after_public'],
                 f'{opponent_name} Reactions to {player_name} Bluff Attempts - Post-flop (Statistics-based)',
                 f'{opponent_name} Action', 'Count', color=reaction_color)

    # 3. Statistical bluff attempts by rank
    if data['statistical_bluff_attempts_by_rank']:
        plot_bar(data['statistical_bluff_attempts_by_rank'],
                 f'{player_name} Statistical Bluff ATTEMPTS by Card Rank',
                 'Rank', 'Attempt Count', color='red')

    # 4. Statistical bluff successes by rank
    if data['statistical_bluff_successes_by_rank']:
        plot_bar(data['statistical_bluff_successes_by_rank'],
                 f'{player_name} Statistical Bluff SUCCESSES by Card Rank',
                 'Rank', 'Success Count', color='green')

    # 5. Statistical bluff attempts by rank group
    if data['statistical_bluff_attempts_by_rank_group']:
        plot_bar(data['statistical_bluff_attempts_by_rank_group'],
                 f'{player_name} Statistical Bluff ATTEMPTS by Rank Group',
                 'Rank Group', 'Attempt Count', color='darkred')

    # 6. Statistical bluff successes by rank group
    if data['statistical_bluff_successes_by_rank_group']:
        plot_bar(data['statistical_bluff_successes_by_rank_group'],
                 f'{player_name} Statistical Bluff SUCCESSES by Rank Group',
                 'Rank Group', 'Success Count', color='darkgreen')

    # 7. Most attempted statistical bluff hands
    if data['statistical_bluff_attempts_by_hand']:
        plot_bar(data['statistical_bluff_attempts_by_hand'],
                 f'Top 15 {player_name} Statistical Bluff ATTEMPT Hands',
                 'Hand', 'Attempt Count', color='red')

    # 8. Most successful statistical bluff hands
    if data['statistical_bluff_successes_by_hand']:
        plot_bar(data['statistical_bluff_successes_by_hand'],
                 f'Top 15 {player_name} Statistical Bluff SUCCESS Hands',
                 'Hand', 'Success Count', color='green')

    # 9. Success rate by hand (only for hands with multiple attempts)
    success_rates_by_hand = {}
    for hand in data['statistical_bluff_attempts_by_hand']:
        attempts = data['statistical_bluff_attempts_by_hand'][hand]
        successes = data['statistical_bluff_successes_by_hand'].get(hand, 0)
        if attempts >= 3:  # Only show hands with at least 3 attempts
            success_rates_by_hand[hand] = successes / attempts

    if success_rates_by_hand:
        plot_bar(success_rates_by_hand,
                 f'{player_name} Statistical Bluff Success Rate by Hand (min 3 attempts)',
                 'Hand', 'Success Rate', color='purple')

    # 10. Statistical bluff attempt outcomes distribution
    if data['statistical_bluff_outcomes'] and sum(data['statistical_bluff_outcomes'].values()) > 0:
        outcomes = list(data['statistical_bluff_outcomes'].keys())
        counts = list(data['statistical_bluff_outcomes'].values())

        plt.figure(figsize=(12, 6))
        colors = ['green', 'orange', 'red', 'gray', 'lightgreen', 'pink']
        bars = plt.bar(outcomes, counts, color=colors, alpha=0.7, edgecolor='black')

        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{count}', ha='center', va='bottom')

        plt.title(f'{player_name} Statistical Bluff Attempt Outcomes Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Outcome', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        wandb.log({f'{player_name} Statistical Bluff Attempt Outcomes Distribution': wandb.Image(plt)})
        plt.close()

    # 11. Success rate comparison by rank
    if data['statistical_bluff_attempts_by_rank'] and data['statistical_bluff_successes_by_rank']:

        all_possible_ranks = ['2', '3', '4', '5', '6', '7', '8', '9']
        ranks = [rank for rank in all_possible_ranks if rank in data['statistical_bluff_attempts_by_rank']]
        attempts = [data['statistical_bluff_attempts_by_rank'].get(rank, 0) for rank in ranks]
        successes = [data['statistical_bluff_successes_by_rank'].get(rank, 0) for rank in ranks]

        plt.figure(figsize=(12, 6))
        x = range(len(ranks))
        width = 0.35

        bars1 = plt.bar([i - width / 2 for i in x], attempts, width, label='Attempts', color='red', alpha=0.7)
        bars2 = plt.bar([i + width / 2 for i in x], successes, width, label='Successes', color='green', alpha=0.7)

        plt.title(f'{player_name} Statistical Bluff Attempts vs Successes by Rank', fontsize=14, fontweight='bold')
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(x, ranks)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        wandb.log({f'{player_name} Statistical Bluff Attempts vs Successes by Rank': wandb.Image(plt)})
        plt.close()

    # Log summary stats to W&B
    summary_stats = {
        'Total Games': data['total_games'],
        f'{player_name} Total Actions': data['player_total_actions'],
        'Total Statistical Bluff Attempts': data['total_statistical_bluff_attempts'],
        'Statistical Bluff Attempt Rate': round(data['statistical_bluff_attempt_rate'], 3),
        'Total Statistical Bluff Successes': data['total_statistical_bluff_successes'],
        'Statistical Bluff Success Rate': round(data['statistical_bluff_success_rate'], 3),
        'Immediate Success Rate': round(data['immediate_successes'] / data['total_statistical_bluff_attempts'], 3) if
        data['total_statistical_bluff_attempts'] > 0 else 0,
        'Lucky Win Rate': round(data['lucky_wins'] / data['total_statistical_bluff_attempts'], 3) if data[
                                                                                                         'total_statistical_bluff_attempts'] > 0 else 0,
        'Total Positive Outcome Rate': round(data['total_positive_outcomes'] / data['total_statistical_bluff_attempts'],
                                             3) if data['total_statistical_bluff_attempts'] > 0 else 0,
    }

    wandb.log(summary_stats)
    wandb.finish()

    print(f"\n=== W&B LOGGED SUMMARY ===")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")


def run_statistical_analysis_both_players(log_path):
    """Run statistical bluff analysis for both players"""

    # Analyze DQN (Player 0)
    print("=" * 80)
    print("ANALYZING DQN (Player 0) - Statistical Bluffs")
    print("=" * 80)
    dqn_data = analyze_statistical_bluffs_52card(log_path, player_id=0, player_name="DQN")
    create_comparable_visualizations(dqn_data, "DQN")

    # Analyze CFR (Player 1)
    print("\n" + "=" * 80)
    print("ANALYZING CFR (Player 1) - Statistical Bluffs")
    print("=" * 80)
    cfr_data = analyze_statistical_bluffs_52card(log_path, player_id=1, player_name="CFR")
    create_comparable_visualizations(cfr_data, "CFR")

    return dqn_data, cfr_data


# Main execution
if __name__ == "__main__":
    log_path = r'C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Evaluation_100K\evaluation_game_logs_all_100K.jsonl'

    dqn_results, cfr_results = run_statistical_analysis_both_players(log_path)

    print("\n" + "=" * 80)
    print("STATISTICAL BLUFF ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"DQN Statistical Bluff Success Rate: {dqn_results['statistical_bluff_success_rate']:.1%}")
    print(f"CFR Statistical Bluff Success Rate: {cfr_results['statistical_bluff_success_rate']:.1%}")

