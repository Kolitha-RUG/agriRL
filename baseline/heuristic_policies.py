# heuristic_policies.py

ACTION_HARVEST = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE = 2
ACTION_REST = 3


def _get_human(env, agent_id):
    # agent_id is likely like "human_0"
    try:
        idx = int(str(agent_id).split("_")[-1])
        return env.humans[idx]
    except Exception:
        pass

    # fallback: match via possible env.agents ordering
    if hasattr(env, "agents") and agent_id in env.agents:
        idx = env.agents.index(agent_id)
        return env.humans[idx]

    raise ValueError(f"Could not resolve human for agent_id={agent_id}")


def _get_vine(env, human):
    return env.vines[human.assigned_vine]


def _can_enqueue(env, vine):
    if hasattr(env, "max_backlog") and hasattr(vine, "queued_boxes"):
        return vine.queued_boxes < env.max_backlog
    return True


def always_enqueue(agent_id, obs, env, threshold=0.7):
    h = _get_human(env, agent_id)
    vine = _get_vine(env, h)

    if h.fatigue >= threshold:
        return ACTION_REST

    if h.has_box:
        return ACTION_ENQUEUE

    if vine.boxes_remaining > 0:
        return ACTION_HARVEST

    return ACTION_REST


def always_transport(agent_id, obs, env, threshold=0.7):
    h = _get_human(env, agent_id)
    vine = _get_vine(env, h)

    if h.fatigue >= threshold:
        return ACTION_REST

    if h.has_box:
        return ACTION_TRANSPORT

    if vine.boxes_remaining > 0:
        return ACTION_HARVEST

    return ACTION_REST


def fatigue_threshold(agent_id, obs, env, threshold=0.7):
    h = _get_human(env, agent_id)
    vine = _get_vine(env, h)

    if h.fatigue >= threshold:
        return ACTION_REST

    if h.has_box:
        return ACTION_TRANSPORT

    if vine.boxes_remaining > 0:
        return ACTION_HARVEST

    return ACTION_REST


def enqueue_when_high_fatigue(agent_id, obs, env, threshold=0.7):
    h = _get_human(env, agent_id)
    vine = _get_vine(env, h)

    if h.has_box:
        if h.fatigue >= threshold:
            if _can_enqueue(env, vine):
                return ACTION_ENQUEUE
            return ACTION_REST
        return ACTION_TRANSPORT

    if vine.boxes_remaining > 0:
        return ACTION_HARVEST

    return ACTION_REST


POLICIES = {
    "always_enqueue": always_enqueue,
    "always_transport": always_transport,
    "fatigue_threshold": fatigue_threshold,
    "enqueue_when_high_fatigue": enqueue_when_high_fatigue,
}