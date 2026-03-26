def value_iteration_step(values, transitions, rewards, gamma):
    num_states = len(values)
    new_values = []

    for s in range(num_states):
        q_values = []
        for a in range(len(transitions[s])):
            expected = 0.0
            for s_prime in range(num_states):
                expected += transitions[s][a][s_prime] * values[s_prime]

            q = rewards[s][a] + gamma * expected
            q_values.append(q)

        new_values.append(max(q_values))

    return new_values