import argparse
from collections import Counter

from agents.simple_agents import HeuristicAgent, RandomAgent
from envs.connect_four_env import ConnectFourEnv


def play_episode(env, agents):
    obs, _ = env.reset()
    done = False
    last_info = {}
    while not done:
        current_agent = agents[env.current_player]
        action = current_agent.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_info = info
    return last_info


def run_eval(episodes: int, use_heuristic: bool):
    env = ConnectFourEnv()
    agent_a = HeuristicAgent() if use_heuristic else RandomAgent()
    agent_b = RandomAgent()
    agents = [agent_a, agent_b]

    results = Counter()
    for _ in range(episodes):
        info = play_episode(env, agents)
        if "winner" in info:
            results[f"player_{info['winner'] + 1}_wins"] += 1
        elif info.get("draw"):
            results["draws"] += 1
        elif info.get("illegal_move"):
            results["illegal_moves"] += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="Self-play evaluation for Connect Four.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic agent as player 1 (else random vs random).",
    )
    args = parser.parse_args()

    results = run_eval(args.episodes, args.heuristic)
    print(f"Ran {args.episodes} episodes. Results: {dict(results)}")


if __name__ == "__main__":
    main()
