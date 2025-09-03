import os
import torch
import pickle
import json
import wandb
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed, reorganize
from rlcard.envs.registration import register
import numpy as np
import collections

# Import custom environment
from BNAIC_paper_files.custom_leduc_rlcard.leducholdem import LeducholdemEnv

# Register the custom environment
register(env_id="custom-leduc-holdem",
         entry_point="BNAIC_paper_files.custom_leduc_rlcard.leducholdem:LeducholdemEnv")

# === Config and Paths ===
SAVE_DIR = r'C:\Users\zaket\PycharmProjects\Thesis\BNAIC_paper_results\simultaneous_Training_100K'
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_DQN_PATH = os.path.join(SAVE_DIR, 'dqn_simultaneous_100K.pt')
SAVE_CFR_PATH = os.path.join(SAVE_DIR, 'cfr_simultaneous_100K.pkl')

wandb.init(project='BNAIC-simultaneous-training-100K', name='Simultaneous_DQN_CFR_52card_100K')
config = {
    "env": "custom-leduc-holdem-52card",
    "train_episodes": 100_000,
    "eval_interval": 5_000,
    "eval_games": 2_000,
    "mlp_layers": [256, 256],
    "learning_rate": 0.00005,
    "batch_size": 64,
    "epsilon_decay_steps": 50_000,
    "epsilon_end": 0.05,
    "memory_init_size": 1000,
    "replay_memory_size": 100_000,
    "iterations_per_episode": 10,
    "seed": 42,
    "deck_size": 52,
    "state_dim": 156,
}



# === CFR Wrapper (for gameplay only) ===
class CFRWrapper:
    def __init__(self, cfr_agent):
        self.cfr = cfr_agent
        self.env = cfr_agent.env
        self.use_raw = False

    def step(self, state):
        obs = state['obs'].tobytes()
        legal_actions = list(state['legal_actions'].keys())
        if obs not in self.cfr.average_policy:
            action_probs = [1 / len(legal_actions)] * self.env.num_actions
        else:
            raw_probs = self.cfr.average_policy[obs]
            action_probs = [raw_probs[a] if a in legal_actions else 0 for a in range(self.env.num_actions)]
            total = sum(action_probs)
            action_probs = [p / total if total > 0 else 1 / len(legal_actions) for p in action_probs]
        return torch.multinomial(torch.tensor(action_probs), 1).item()

    def eval_step(self, state):
        action = self.step(state)
        return action, {"probs": {}}


# === CFR Agent That Trains Against Live DQN ===
def zero_array_4():
    return np.zeros(4)


class CFRAgainstDQNAgent:
    def __init__(self, env, player_id, opponent_agent, model_path):
        self.env = env
        self.player_id = player_id
        self.opponent_agent = opponent_agent
        self.model_path = model_path
        self.use_raw = False

        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(zero_array_4)
        self.regrets = collections.defaultdict(zero_array_4)
        self.iteration = 0

    def regret_matching(self, obs):
        regret = self.regrets[obs]
        pos_regret = np.maximum(regret, 0)
        total = np.sum(pos_regret)
        return pos_regret / total if total > 0 else np.ones(self.env.num_actions) / self.env.num_actions

    def traverse_tree(self, probs):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        obs = state['obs'].tobytes()
        legal_actions = list(state['legal_actions'].keys())

        if current_player != self.player_id:
            action, _ = self.opponent_agent.eval_step(state)
            self.env.step(action)
            utility = self.traverse_tree(probs)
            self.env.step_back()
            return utility

        strategy = self.regret_matching(obs)
        action_utils = np.zeros(self.env.num_actions)
        node_util = 0

        for action in legal_actions:
            prob = strategy[action]
            new_probs = probs.copy()
            new_probs[current_player] *= prob
            self.env.step(action)
            utility = self.traverse_tree(new_probs)
            self.env.step_back()
            action_utils[action] = utility[self.player_id]
            node_util += prob * utility[self.player_id]

        cf_prob = np.prod(probs[:current_player]) * np.prod(probs[current_player + 1:])
        for action in legal_actions:
            regret = cf_prob * (action_utils[action] - node_util)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += probs[current_player] * strategy[action]

        self.policy[obs] = strategy
        return np.array([node_util if i == self.player_id else 0 for i in range(self.env.num_players)])

    def save(self):
        data = {
            'policy': dict(self.policy),
            'average_policy': dict(self.average_policy),
            'regrets': dict(self.regrets),
            'iteration': self.iteration
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)


# === Main Training Loop ===
def train():
    set_seed(config['seed'])

    # Use custom environment directly
    env = LeducholdemEnv(config={'seed': config['seed'], 'allow_step_back': True})

    # Initialize game and check judger
    env.reset()
    print("=" * 70)
    print("100K SIMULTANEOUS TRAINING - 52-CARD LEDUC HOLD'EM")
    print("=" * 70)
    print(f"Using environment: {env.name}")
    print(f"Deck size: {len(env.game.dealer.deck)} cards")
    print(f"State shape: {env.state_shape[0]} dimensions")
    print(f"Training episodes: {config['train_episodes']:,}")
    print(f"Evaluation every: {config['eval_interval']:,} episodes")
    print(f"Judger module: {env.game.judger.__class__.__module__}")
    if hasattr(env.game.judger, 'RANK_ORDER'):
        print(f"RANK_ORDER: {list(env.game.judger.RANK_ORDER.keys())}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=config['mlp_layers'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay_steps=config['epsilon_decay_steps'],
        replay_memory_init_size=config['memory_init_size'],
        replay_memory_size=config['replay_memory_size'],
        device=device
    )

    print(f"DQN Configuration:")
    print(f"  Network: {config['mlp_layers']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epsilon decay: {config['epsilon_decay_steps']:,} steps")
    print(f"  Replay memory: {config['replay_memory_size']:,}")
    print(f"  Device: {device}")
    print()

    cfr_agent = CFRAgainstDQNAgent(env, player_id=1, opponent_agent=dqn_agent, model_path=SAVE_CFR_PATH)
    env.set_agents([dqn_agent, CFRWrapper(cfr_agent)])

    for episode in range(config['train_episodes']):
            # CFR training iterations
            for _ in range(config['iterations_per_episode']):
                env.reset()
                cfr_agent.traverse_tree(np.ones(env.num_players))
                cfr_agent.iteration += 1

            if episode % 1000 == 0:
                total_regret = sum(np.sum(np.abs(r)) for r in cfr_agent.regrets.values())
                print(f"[Episode {episode:,}] CFR iterations: {cfr_agent.iteration:,}, "
                      f"States in policy: {len(cfr_agent.average_policy):,}")

                wandb.log({
                    "cfr_iterations": cfr_agent.iteration,
                    "cfr_states_seen": len(cfr_agent.average_policy),
                    "cfr_total_regret": total_regret,
                    "episode": episode
                })

            # DQN training from actual gameplay
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                if ts:
                    dqn_agent.feed(ts)

            wandb.log({"dqn_reward": payoffs[0], "episode": episode})

            if episode % config['eval_interval'] == 0:
                dqn_rewards, cfr_rewards, wins = [], [], [0, 0, 0]
                for _ in range(config['eval_games']):
                    _, payoffs = env.run(is_training=False)
                    dqn_rewards.append(payoffs[0])
                    cfr_rewards.append(payoffs[1])
                    if payoffs[0] > payoffs[1]:
                        wins[0] += 1
                    elif payoffs[1] > payoffs[0]:
                        wins[1] += 1
                    else:
                        wins[2] += 1

                dqn_wr = wins[0] / config['eval_games']
                cfr_wr = wins[1] / config['eval_games']
                draw_rate = wins[2] / config['eval_games']

                wandb.log({
                    "dqn_win_rate_vs_cfr": dqn_wr,
                    "cfr_win_rate_vs_dqn": cfr_wr,
                    "draw_rate": draw_rate,
                    "dqn_reward_eval": np.mean(dqn_rewards),
                    "dqn_reward_std": np.std(dqn_rewards),
                    "cfr_reward_eval": np.mean(cfr_rewards),
                    "cfr_reward_std": np.std(cfr_rewards),
                    "training_progress": episode / config['train_episodes'],  # ADDED: Progress tracking
                    "episode": episode
                })

                print(f"[Ep {episode:,}/{config['train_episodes']:,}] "
                      f"DQN WR: {dqn_wr:.3f} | CFR WR: {cfr_wr:.3f} | "
                      f"Draws: {draw_rate:.3f} | Progress: {episode/config['train_episodes']:.1%}")


    # Save models
    torch.save(dqn_agent.q_estimator.qnet.state_dict(), SAVE_DQN_PATH)
    cfr_agent.save()

    print(f"\n✅ 100K SIMULTANEOUS TRAINING COMPLETE!")
    print(f"✅ Saved DQN model to {SAVE_DQN_PATH}")
    print(f"✅ Saved CFR model to {SAVE_CFR_PATH}")

    print("\n" + "=" * 70)
    print("FINAL TRAINING SUMMARY")
    print("=" * 70)
    print(f"Environment: 52-card Leduc Hold'em")
    print(f"Total episodes: {config['train_episodes']:,}")
    print(f"Total CFR iterations: {cfr_agent.iteration:,}")
    print(f"Total states in CFR policy: {len(cfr_agent.average_policy):,}")
    print(f"DQN network size: {config['mlp_layers']}")
    print(f"Expected draw rate: 0% (deterministic judger)")
    print("=" * 70)


if __name__ == '__main__':
    train()
