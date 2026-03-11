# evaluation_utils.py

def run_episode(env, policy_fn, threshold=0.7):
    obs, info = env.reset()

    done = False
    infos = {}

    while not done:
        actions = {}

        for agent_id, agent_obs in obs.items():
            actions[agent_id] = policy_fn(agent_id, agent_obs, env, threshold)

        obs, rewards, terminated, truncated, infos = env.step(actions)
        done = terminated["__all__"] or truncated["__all__"]

    # Your env stores episode_summary inside each agent info
    for agent_id, agent_info in infos.items():
        if isinstance(agent_info, dict) and "episode_summary" in agent_info:
            summary = agent_info["episode_summary"]
            if summary:   # make sure it's not empty
                return summary

    raise KeyError("Could not find non-empty episode_summary in final infos.")