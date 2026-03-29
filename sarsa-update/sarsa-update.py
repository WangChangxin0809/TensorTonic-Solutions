def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    new_q_table = q_table.copy()

    delta = reward + gamma * q_table[next_state][next_action] - q_table[state][action]

    new_q_table[state][action] = q_table[state][action] + alpha * delta

    return new_q_table
