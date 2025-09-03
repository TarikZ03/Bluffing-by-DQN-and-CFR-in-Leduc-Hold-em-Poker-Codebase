import json
import wandb
import matplotlib.pyplot as plt
from collections import Counter
import os
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === Configs ===
LOG_PATH = r'C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Evaluation_100K\evaluation_game_logs_all_100K.jsonl'
CFR_PLAYER_ID = 1  # CFR is the bluffer
PROJECT_NAME = 'BNAIC-bluff-analysis-52card'
RUN_NAME = 'DQN_Reaction_to_CFR_Bluffs_52Card_Leduc_CLEAR_LABELS'

wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# === Constants for 52-Card Custom Leduc ===
RANK_ORDER = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
              'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_ORDER = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
action_id_to_name = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}


# === Helper Functions ===
def card_score(card_str):
    if len(card_str) >= 2:
        suit = card_str[0]
        rank = card_str[1]
        return RANK_ORDER.get(rank, 0) * 4 + SUIT_ORDER.get(suit, 0)
    return 0


def hand_strength_category(hand, public_card=None):
    hand_score = card_score(hand)
    if public_card and len(public_card) >= 2:
        if hand[1] == public_card[1]:  # Same rank = pair
            return hand_score + 1000
    return hand_score


def is_bluff_attempt(hand, action_index, public_card=None):

    if action_index != 1:  # Not a raise
        return False
    strength = hand_strength_category(hand, public_card)
    return strength < 32  # Less than 8s (7 or lower without pair)


def get_hand_category(hand):
    if len(hand) >= 2:
        rank = hand[1]
        suit = hand[0]
        return f"{rank}{suit}"
    return hand


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


# === Initialize Counters ===
total_games = 0
cfr_total_actions = 0

total_bluff_attempts = 0
total_bluff_successes = 0  # Only when opponent folds

# Detailed outcome tracking
bluff_attempt_outcomes = {
    'opponent_folded': 0,  # SUCCESS: Opponent folded to our bluff
    'opponent_called': 0,  # FAILED: Opponent called our bluff
    'opponent_raised': 0,  # FAILED: Opponent re-raised our bluff
    'opponent_checked': 0,  # FAILED: Opponent checked (shouldn't happen after raise)
    'showdown_won': 0,  # LUCKY: We bluffed, got called, but won anyway
    'showdown_lost': 0  # EXPECTED: We bluffed, got called, and lost
}

# Tracking by hand - ATTEMPTS
bluff_attempts_by_hand = Counter()
bluff_attempts_by_rank = Counter()
bluff_attempts_by_rank_group = Counter()

# Tracking by hand - SUCCESSES (only when opponent folded)
bluff_successes_by_hand = Counter()
bluff_successes_by_rank = Counter()
bluff_successes_by_rank_group = Counter()

# Opponent reactions to bluff attempts
opponent_reactions = Counter()
reactions_before_public = Counter()
reactions_after_public = Counter()
reaction_by_bluff_hand = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}

# === Main Analysis Loop ===
print(f"Loading logs from: {LOG_PATH}")
if not os.path.exists(LOG_PATH):
    print(f"ERROR: Log file not found at {LOG_PATH}")
    exit(1)

with open(LOG_PATH, 'r') as file:
    for line_num, line in enumerate(file):
        try:
            data = json.loads(line)
            total_games += 1
            log = data.get("log", [])
            payoffs = data.get("payoffs", [0, 0])

            for i, entry in enumerate(log):
                player_id = entry['player_id']

                if player_id == CFR_PLAYER_ID:
                    cfr_total_actions += 1

                    hand = entry.get('hand', '')
                    action_index = entry.get('action_taken', -1)
                    public_card = entry.get('public_card', None)

                    # Check for BLUFF ATTEMPT
                    if hand and is_bluff_attempt(hand, action_index, public_card):
                        total_bluff_attempts += 1

                        # Track attempt details
                        hand_cat = get_hand_category(hand)
                        rank_group = get_rank_group(hand)

                        bluff_attempts_by_hand[hand_cat] += 1
                        bluff_attempts_by_rank[hand[1]] += 1
                        bluff_attempts_by_rank_group[rank_group] += 1

                        # Look for opponent's reaction (DQN)
                        opponent_reaction = None
                        opponent_action_index = None
                        reaction_context = None

                        for j in range(i + 1, len(log)):
                            next_entry = log[j]
                            if next_entry['player_id'] != CFR_PLAYER_ID:  # DQN's turn
                                opponent_action_index = next_entry.get('action_taken', -1)
                                opponent_reaction = action_id_to_name.get(opponent_action_index, 'UNKNOWN')
                                reaction_context = next_entry.get('public_card', None)
                                break

                        # Track reaction and determine success
                        if opponent_reaction and opponent_reaction != 'UNKNOWN':
                            opponent_reactions[opponent_reaction] += 1
                            reaction_by_bluff_hand[opponent_action_index][hand_cat] += 1

                            # Context tracking
                            if reaction_context:
                                reactions_after_public[opponent_reaction] += 1
                            else:
                                reactions_before_public[opponent_reaction] += 1

                            # Determine outcome
                            if opponent_reaction == 'fold':
                                # BLUFF SUCCESS!
                                total_bluff_successes += 1
                                bluff_attempt_outcomes['opponent_folded'] += 1

                                # Track successful bluff by hand
                                bluff_successes_by_hand[hand_cat] += 1
                                bluff_successes_by_rank[hand[1]] += 1
                                bluff_successes_by_rank_group[rank_group] += 1

                            elif opponent_reaction == 'call':
                                bluff_attempt_outcomes['opponent_called'] += 1
                                # Check final showdown result
                                if payoffs[CFR_PLAYER_ID] > 0:
                                    bluff_attempt_outcomes['showdown_won'] += 1
                                else:
                                    bluff_attempt_outcomes['showdown_lost'] += 1

                            elif opponent_reaction == 'raise':
                                bluff_attempt_outcomes['opponent_raised'] += 1
                            elif opponent_reaction == 'check':
                                bluff_attempt_outcomes['opponent_checked'] += 1

                        if total_bluff_attempts <= 5:
                            success = "SUCCESS" if opponent_reaction == 'fold' else "FAILED"
                            print(
                                f"DEBUG Attempt #{total_bluff_attempts}: {hand} -> DQN {opponent_reaction} -> {success}")

        except json.JSONDecodeError:
            print(f"Warning: Could not parse line {line_num + 1}")
            continue

# === Calculate Key Metrics ===
bluff_attempt_rate = total_bluff_attempts / cfr_total_actions if cfr_total_actions > 0 else 0
bluff_success_rate = total_bluff_successes / total_bluff_attempts if total_bluff_attempts > 0 else 0

immediate_successes = bluff_attempt_outcomes['opponent_folded']
lucky_wins = bluff_attempt_outcomes['showdown_won']
total_positive_outcomes = immediate_successes + lucky_wins

# === Print Results ===
print("\n" + "=" * 80)
print("CFR BLUFF ANALYSIS - ATTEMPTS vs SUCCESSES")
print("=" * 80)

print(f"\n=== BASIC METRICS ===")
print(f"Total Games: {total_games}")
print(f"CFR Total Actions: {cfr_total_actions}")
print(f"")
print(f"BLUFF ATTEMPTS (raise with 7 or lower): {total_bluff_attempts}")
print(f"Bluff Attempt Rate: {bluff_attempt_rate:.3f} ({bluff_attempt_rate * 100:.1f}%)")
print(f"")
print(f"BLUFF SUCCESSES (DQN folded): {total_bluff_successes}")
print(f"Bluff Success Rate: {bluff_success_rate:.3f} ({bluff_success_rate * 100:.1f}%)")

print(f"\n=== DQN REACTIONS TO BLUFF ATTEMPTS ===")
total_reactions = sum(opponent_reactions.values())
for reaction, count in opponent_reactions.most_common():
    percentage = (count / total_reactions) * 100 if total_reactions > 0 else 0
    print(f"  {reaction}: {count} ({percentage:.1f}%)")

print(f"\n=== REACTIONS BY CONTEXT ===")
print("Before Public Card (Pre-flop):")
total_pre = sum(reactions_before_public.values())
for reaction, count in reactions_before_public.most_common():
    percentage = (count / total_pre) * 100 if total_pre > 0 else 0
    print(f"  {reaction}: {count} ({percentage:.1f}%)")

print("After Public Card (Post-flop):")
total_post = sum(reactions_after_public.values())
for reaction, count in reactions_after_public.most_common():
    percentage = (count / total_post) * 100 if total_post > 0 else 0
    print(f"  {reaction}: {count} ({percentage:.1f}%)")

print(f"\n=== DETAILED BLUFF ATTEMPT OUTCOMES ===")
for outcome, count in bluff_attempt_outcomes.items():
    percentage = (count / total_bluff_attempts) * 100 if total_bluff_attempts > 0 else 0
    print(f"  {outcome.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

print(f"\n=== SUCCESS BREAKDOWN ===")
print(
    f"  Immediate successes (DQN folded): {immediate_successes}/{total_bluff_attempts} ({immediate_successes / total_bluff_attempts * 100:.1f}%)")
print(
    f"  Lucky wins (called but won): {lucky_wins}/{total_bluff_attempts} ({lucky_wins / total_bluff_attempts * 100:.1f}%)")
print(
    f"  Total positive outcomes: {total_positive_outcomes}/{total_bluff_attempts} ({total_positive_outcomes / total_bluff_attempts * 100:.1f}%)")

print(f"\n=== BLUFF ATTEMPTS BY RANK GROUP ===")
for group, count in bluff_attempts_by_rank_group.most_common():
    percentage = (count / total_bluff_attempts) * 100 if total_bluff_attempts > 0 else 0
    print(f"  {group}: {count} ({percentage:.1f}%)")

print(f"\n=== BLUFF SUCCESSES BY RANK GROUP ===")
for group, count in bluff_successes_by_rank_group.most_common():
    success_rate = count / bluff_attempts_by_rank_group[group] * 100 if bluff_attempts_by_rank_group[group] > 0 else 0
    print(f"  {group}: {count} successes / {bluff_attempts_by_rank_group[group]} attempts ({success_rate:.1f}%)")

print(f"\n=== BLUFF ATTEMPTS BY INDIVIDUAL RANK ===")
for rank, count in bluff_attempts_by_rank.most_common():
    percentage = (count / total_bluff_attempts) * 100 if total_bluff_attempts > 0 else 0
    print(f"  {rank}: {count} ({percentage:.1f}%)")

print(f"\n=== TOP 10 MOST ATTEMPTED BLUFF HANDS ===")
for hand, attempt_count in bluff_attempts_by_hand.most_common(10):
    success_count = bluff_successes_by_hand.get(hand, 0)
    success_rate = success_count / attempt_count * 100 if attempt_count > 0 else 0
    print(f"  {hand}: {attempt_count} attempts, {success_count} successes ({success_rate:.1f}%)")

print(f"\n=== TOP 10 MOST SUCCESSFUL BLUFF HANDS ===")
for hand, success_count in bluff_successes_by_hand.most_common(10):
    attempt_count = bluff_attempts_by_hand[hand]
    success_rate = success_count / attempt_count * 100 if attempt_count > 0 else 0
    print(f"  {hand}: {success_count} successes / {attempt_count} attempts ({success_rate:.1f}%)")

# === Log to W&B ===
summary_stats = {
    'Total Games': total_games,
    'CFR Total Actions': cfr_total_actions,
    'Total Bluff Attempts': total_bluff_attempts,
    'Bluff Attempt Rate': round(bluff_attempt_rate, 3),
    'Total Bluff Successes': total_bluff_successes,
    'Bluff Success Rate': round(bluff_success_rate, 3),
    'Immediate Success Rate': round(immediate_successes / total_bluff_attempts, 3) if total_bluff_attempts > 0 else 0,
    'Lucky Win Rate': round(lucky_wins / total_bluff_attempts, 3) if total_bluff_attempts > 0 else 0,
    'Total Positive Outcome Rate': round(total_positive_outcomes / total_bluff_attempts,
                                         3) if total_bluff_attempts > 0 else 0,
}

wandb.log(summary_stats)


# === PLOTTING FUNCTIONS ===
def plot_bar(data, title, xlabel, ylabel, color='blue', figsize=(10, 6)):
    if not data:
        print(f"No data to plot for: {title}")
        return

    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:15]  # Top 15
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


# === VISUALIZATIONS ===

# 1. DQN reactions to bluff attempts
if opponent_reactions:
    plot_bar(opponent_reactions, 'DQN Reactions to CFR Bluff ATTEMPTS', 'DQN Action', 'Count')

# 2. Pre/Post flop context
if reactions_before_public:
    plot_bar(reactions_before_public, 'DQN Reactions to Bluff ATTEMPTS (Pre-flop)', 'DQN Action', 'Count')
if reactions_after_public:
    plot_bar(reactions_after_public, 'DQN Reactions to Bluff ATTEMPTS (Post-flop)', 'DQN Action', 'Count')

# 3. Bluff attempts by rank
if bluff_attempts_by_rank:
    plot_bar(bluff_attempts_by_rank, 'CFR Bluff ATTEMPTS by Card Rank', 'Rank', 'Attempt Count', color='blue')

# 4. Bluff successes by rank
if bluff_successes_by_rank:
    plot_bar(bluff_successes_by_rank, 'CFR Bluff SUCCESSES by Card Rank', 'Rank', 'Success Count', color='green')

# 5. Bluff attempts by rank group
if bluff_attempts_by_rank_group:
    plot_bar(bluff_attempts_by_rank_group, 'CFR Bluff ATTEMPTS by Rank Group', 'Rank Group', 'Attempt Count',
             color='darkblue')

# 6. Bluff successes by rank group
if bluff_successes_by_rank_group:
    plot_bar(bluff_successes_by_rank_group, 'CFR Bluff SUCCESSES by Rank Group', 'Rank Group', 'Success Count',
             color='darkgreen')

# 7. Most attempted bluff hands
if bluff_attempts_by_hand:
    plot_bar(bluff_attempts_by_hand, 'Top 15 CFR Bluff ATTEMPT Hands', 'Hand', 'Attempt Count', color='blue')

# 8. Most successful bluff hands
if bluff_successes_by_hand:
    plot_bar(bluff_successes_by_hand, 'Top 15 CFR Bluff SUCCESS Hands', 'Hand', 'Success Count', color='green')

# 9. Success rate by hand (only for hands with multiple attempts)
success_rates_by_hand = {}
for hand in bluff_attempts_by_hand:
    attempts = bluff_attempts_by_hand[hand]
    successes = bluff_successes_by_hand.get(hand, 0)
    if attempts >= 3:  # Only show hands with at least 3 attempts
        success_rates_by_hand[hand] = successes / attempts

if success_rates_by_hand:
    plot_bar(success_rates_by_hand, 'Bluff Success Rate by Hand (min 3 attempts)', 'Hand', 'Success Rate',
             color='purple')

# 10. Bluff attempt outcomes distribution
if bluff_attempt_outcomes and sum(bluff_attempt_outcomes.values()) > 0:
    outcomes = list(bluff_attempt_outcomes.keys())
    counts = list(bluff_attempt_outcomes.values())

    plt.figure(figsize=(12, 6))
    colors = ['green', 'orange', 'red', 'gray', 'lightgreen', 'pink']
    bars = plt.bar(outcomes, counts, color=colors, alpha=0.7, edgecolor='black')

    for bar, count in zip(bars, counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{count}', ha='center', va='bottom')

    plt.title('CFR Bluff Attempt Outcomes Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Outcome', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    wandb.log({'CFR Bluff Attempt Outcomes Distribution': wandb.Image(plt)})
    plt.close()

# 11. Success rate comparison by rank
if bluff_attempts_by_rank and bluff_successes_by_rank:
    ranks = sorted(bluff_attempts_by_rank.keys())
    attempts = [bluff_attempts_by_rank[rank] for rank in ranks]
    successes = [bluff_successes_by_rank.get(rank, 0) for rank in ranks]

    plt.figure(figsize=(12, 6))
    x = range(len(ranks))
    width = 0.35

    bars1 = plt.bar([i - width / 2 for i in x], attempts, width, label='Attempts', color='red', alpha=0.7)
    bars2 = plt.bar([i + width / 2 for i in x], successes, width, label='Successes', color='green', alpha=0.7)

    plt.title('CFR Bluff Attempts vs Successes by Rank', fontsize=14, fontweight='bold')
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, ranks)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    wandb.log({'CFR Bluff Attempts vs Successes by Rank': wandb.Image(plt)})
    plt.close()

# 12. DQN fold reactions by bluffing hand
if reaction_by_bluff_hand[2]:  # Fold reactions
    plot_bar(reaction_by_bluff_hand[2], 'DQN Fold Reactions by CFR Bluff Hand', 'Bluff Hand', 'Fold Count',
             color='green')

wandb.finish()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE WITH CLEAR ATTEMPT/SUCCESS SEPARATION")
print("=" * 80)
print(f"Key Insight: Out of {total_bluff_attempts} bluff attempts, {total_bluff_successes} succeeded")
print(f"This gives a {bluff_success_rate:.1%} success rate for CFR's weak hand raises against DQN")

