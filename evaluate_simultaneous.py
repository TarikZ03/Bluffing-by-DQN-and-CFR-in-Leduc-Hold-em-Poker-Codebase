import os
import json
import pickle
import torch
import numpy as np
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed
from rlcard.envs.registration import register

# Import custom environment
from BNAIC_paper_files.custom_leduc_rlcard.leducholdem import LeducholdemEnv

# Register the custom environment
register(env_id="custom-leduc-holdem",
         entry_point="BNAIC_paper_files.custom_leduc_rlcard.leducholdem:LeducholdemEnv")

NUM_GAMES = 100_000
SEED = 42
SAVE_DIR = r'C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Evaluation_100K'
CFR_MODEL_PATH = r"C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Training_100K\cfr_simultaneous_100K.pkl"
DQN_MODEL_PATH = r"C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Training_100K\dqn_simultaneous_100K.pt"
LOG_ALL_PATH = os.path.join(SAVE_DIR, 'evaluation_game_logs_all_100K.jsonl')
LOG_CFR_PATH = os.path.join(SAVE_DIR, 'evaluation_game_logs_cfr_pov_100K.jsonl')
LOG_DQN_PATH = os.path.join(SAVE_DIR, 'evaluation_game_logs_dqn_pov_100K.jsonl')


# Helper Classes & Functions
class CFRWrapper:
    """Wrapper for CFR agent to work with evaluation"""

    def __init__(self, average_policy, env):
        self.average_policy = average_policy
        self.env = env
        self.use_raw = False

    def step(self, state):
        obs = state['obs'].tobytes()
        legal_actions = list(state['legal_actions'].keys())

        if obs not in self.average_policy:
            # Uniform random for unseen states
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions
        else:
            # Get stored probabilities
            action_probs = self.average_policy[obs].copy()

        # Remove illegal actions
        for a in range(self.env.num_actions):
            if a not in legal_actions:
                action_probs[a] = 0

        # Renormalize
        total = np.sum(action_probs)
        if total > 0:
            action_probs = action_probs / total
        else:
            # Fallback to uniform over legal actions
            action_probs = np.zeros(self.env.num_actions)
            for a in legal_actions:
                action_probs[a] = 1.0 / len(legal_actions)

        # Sample action
        action = np.random.choice(self.env.num_actions, p=action_probs)
        return action

    def eval_step(self, state):
        action = self.step(state)
        return action, {"probs": {}}


def convert_ndarrays(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def load_agents(env):

    print("Loading agents...")

    # Load CFR agent
    if not os.path.exists(CFR_MODEL_PATH):
        raise FileNotFoundError(f"CFR model not found at {CFR_MODEL_PATH}")

    with open(CFR_MODEL_PATH, 'rb') as f:
        cfr_data = pickle.load(f)

    # Extract average policy
    average_policy = cfr_data['average_policy']
    print(f"CFR loaded with {len(average_policy)} states in policy")

    # Create CFR wrapper
    cfr_agent = CFRWrapper(average_policy, env)

    # Load DQN agent
    if not os.path.exists(DQN_MODEL_PATH):
        raise FileNotFoundError(f"DQN model not found at {DQN_MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256],
        device=device
    )

    # Load weights
    dqn_agent.q_estimator.qnet.load_state_dict(
        torch.load(DQN_MODEL_PATH, map_location=device)
    )
    dqn_agent.q_estimator.qnet.eval()

    print("Both agents loaded successfully")
    return dqn_agent, cfr_agent


def evaluate():

    set_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create custom environment
    env = LeducholdemEnv(config={'seed': SEED, 'allow_step_back': False})
    env.reset()

    print("=" * 70)
    print(f"Using environment: {env.name}")
    print(f"Judger module: {env.game.judger.__class__.__module__}")
    print(f"Has RANK_ORDER: {hasattr(env.game.judger, 'RANK_ORDER')}")
    print("=" * 70)

    # Load agents
    dqn_agent, cfr_agent = load_agents(env)
    env.set_agents([dqn_agent, cfr_agent])

    # Statistics tracking
    wins = [0, 0, 0]  # [DQN wins, CFR wins, Draws]
    total_payoffs = [0.0, 0.0]

    unique_hands = set()

    print(f"\nStarting evaluation of {NUM_GAMES} games...")

    # Open log files
    with open(LOG_ALL_PATH, 'w') as all_f, \
            open(LOG_CFR_PATH, 'w') as cfr_f, \
            open(LOG_DQN_PATH, 'w') as dqn_f:

        for game_num in range(1, NUM_GAMES + 1):
            env.reset()
            logs = []

            # Play one game
            while not env.is_over():
                player_id = env.get_player_id()
                state = env.get_state(player_id)

                # Get action from appropriate agent
                if player_id == 0:
                    action, _ = dqn_agent.eval_step(state)
                else:
                    action, _ = cfr_agent.eval_step(state)

                # Log game state
                raw = state.get('raw_obs', {})

                hand = raw.get("hand")
                if hand:
                    unique_hands.add(hand)

                logs.append({
                    "player_id": player_id,
                    "hand": hand,
                    "public_card": raw.get("public_card"),
                    "legal_actions": raw.get("legal_actions"),
                    "action_record": raw.get("action_record"),
                    "action_taken": int(action),
                })

                env.step(action)

            # Get game results
            payoffs = env.get_payoffs()

            # Update statistics
            total_payoffs[0] += payoffs[0]
            total_payoffs[1] += payoffs[1]

            if payoffs[0] > payoffs[1]:
                wins[0] += 1
            elif payoffs[1] > payoffs[0]:
                wins[1] += 1
            else:
                wins[2] += 1

            # Prepare results for logging
            full_result = {
                "game": game_num,
                "log": convert_ndarrays(logs),
                "payoffs": convert_ndarrays(payoffs),
            }

            cfr_result = {
                "game": game_num,
                "log": convert_ndarrays([log for log in logs if log["player_id"] == 1]),
                "payoffs": convert_ndarrays(payoffs),
            }

            dqn_result = {
                "game": game_num,
                "log": convert_ndarrays([log for log in logs if log["player_id"] == 0]),
                "payoffs": convert_ndarrays(payoffs),
            }

            # Write to files
            all_f.write(json.dumps(full_result) + '\n')
            cfr_f.write(json.dumps(cfr_result) + '\n')
            dqn_f.write(json.dumps(dqn_result) + '\n')

            if game_num % 10000 == 0:
                dqn_wr = wins[0] / game_num
                cfr_wr = wins[1] / game_num
                draw_rate = wins[2] / game_num
                print(
                    f"[{game_num}/{NUM_GAMES}] DQN: {dqn_wr:.3f}, CFR: {cfr_wr:.3f}, Draws: {draw_rate:.3f}, Unique hands: {len(unique_hands)}")

    # Final statistics
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total games played: {NUM_GAMES}")
    print(f"DQN wins: {wins[0]} ({wins[0] / NUM_GAMES * 100:.2f}%)")
    print(f"CFR wins: {wins[1]} ({wins[1] / NUM_GAMES * 100:.2f}%)")
    print(f"Draws: {wins[2]} ({wins[2] / NUM_GAMES * 100:.2f}%)")
    print(f"\nAverage payoffs:")
    print(f"  DQN: {total_payoffs[0] / NUM_GAMES:.4f}")
    print(f"  CFR: {total_payoffs[1] / NUM_GAMES:.4f}")
    print(f"\nUnique hands seen: {len(unique_hands)} out of 52 possible")
    print(f"\nLog files saved to:")
    print(f"  ➤ All games: {LOG_ALL_PATH}")
    print(f"  ➤ CFR POV:   {LOG_CFR_PATH}")
    print(f"  ➤ DQN POV:   {LOG_DQN_PATH}")

    if wins[2] > 0:
        print(f"\nWARNING: Found {wins[2]} draws with custom judger!")
        print("This suggests the custom judger might not be working as expected.")
    else:
        print("\n✓ No draws found (expected with deterministic custom judger)")


if __name__ == '__main__':
    evaluate()
