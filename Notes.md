# Reward_DF

    # -----------------------------
    # SIMPLE COOPERATIVE REWARD
    # -----------------------------
    mean_fatigue_t = float(np.mean([h.fatigue for h in self.humans]))

    r_delivery = self.reward_delivery * float(delivered_delta)
    r_fatigue = -self.reward_fatigue_level_penalty * mean_fatigue_t

    team_reward = r_delivery + r_fatigue

    step_rewards = {}

    for agent_id in self.possible_agents:
        step_rewards[agent_id] = float(team_reward)
        self.pending_rewards[agent_id] += float(team_reward)

# Reward_Old

        # Drones: utilization = flying steps
        for d in self.drones:
            if d.busy and d.status in (DRONE_GO_TO_HANDOVER, DRONE_DELIVER):
                self.ep_drone_flying_steps += 1

        fatigue_total = float(sum(fatigue_increase))

        # -----------------------------
        # TEAM REWARD
        # -----------------------------
        backlog_capacity = max(1, len(self.handover_points) * self.max_backlog)
        backlog_norm = float(backlog_total) / backlog_capacity

        fatigue_excess = float(np.mean([max(0.0, h.fatigue - 0.6) for h in self.humans]))

        r_delivery = self.reward_delivery * float(delivered_delta)
        r_fat_inc = -self.reward_fatigue_inc_penalty * fatigue_total
        r_backlog = -self.reward_backlog_penalty * backlog_norm
        r_fat_lvl = -self.reward_fatigue_level_penalty * fatigue_excess

        team_reward = r_delivery + r_fat_inc + r_backlog + r_fat_lvl

        self.ep_r_delivery_sum += float(r_delivery)
        self.ep_r_fatigue_inc_sum += float(r_fat_inc)
        self.ep_r_backlog_sum += float(r_backlog)
        self.ep_r_fatigue_level_sum += float(r_fat_lvl)

        lambda_team = 0.60
        step_rewards = {}

        for i, agent_id in enumerate(self.possible_agents):
            fatigue_excess_i = max(0.0, self.humans[i].fatigue - 0.6)

            local_reward = (
                self.reward_delivery * float(individual_deliveries[i])
                + self.reward_harvest * float(harvest_events[i])
                + self.reward_enqueue * float(enqueue_events[i])
                + self.reward_drone_credit * float(drone_credit_deliveries[i])
                - self.reward_fatigue_inc_penalty * float(fatigue_increase[i])
                - self.reward_fatigue_level_penalty * float(fatigue_excess_i)
            )

            agent_reward = lambda_team * team_reward + (1.0 - lambda_team) * local_reward
            step_rewards[agent_id] = float(agent_reward)
            self.pending_rewards[agent_id] += float(agent_reward)

        # for episode-level logging
        self.ep_reward_sum += float(np.mean(list(step_rewards.values())))
