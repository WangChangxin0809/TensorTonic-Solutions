from collections import defaultdict

import numpy as np


def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    ps = []
    cd_len, rf_len = len(candidate), len(reference)
    bp = np.exp(1 - rf_len / cd_len) if cd_len < rf_len and cd_len else 1.0

    for i in range(max_n):

        ddt_cd = defaultdict(lambda: 0)
        for j in range(cd_len - i):
            ddt_cd[','.join(candidate[j : j+i+1])] += 1

        ddt_rf = defaultdict(lambda: 0)
        for j in range(rf_len - i):
            ddt_rf[','.join(reference[j : j+i+1])] += 1

        accuracy = 0
        for key, val in ddt_cd.items():
            accuracy += min(val, ddt_rf[key])

        if accuracy == 0:
            return 0

        ps.append(np.log(accuracy / (cd_len - i)))

    return np.exp(sum(ps) / len(ps)) * bp