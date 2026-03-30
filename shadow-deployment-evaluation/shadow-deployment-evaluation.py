from math import ceil


def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    log = {
        "promote": False,
        "metrics": {
            "shadow_accuracy": 0.0,
            "production_accuracy": 0.0,
            "accuracy_gain": 0.0,
            "shadow_latency_p95": 0,
            "agreement_rate": 0.0
        }
    }

    latencies = []

    cnt = len(production_log)
    for i in range(cnt):
        production_dict, shadow_dict = production_log[i], shadow_log[i]
        log['metrics']['production_accuracy'] += production_dict['prediction'] == production_dict['actual']
        log['metrics']['shadow_accuracy'] += shadow_dict['prediction'] == shadow_dict['actual']
        log['metrics']['agreement_rate'] += production_dict['prediction'] == shadow_dict['prediction']
        latencies.append(shadow_dict['latency_ms'])

    log['metrics']['shadow_latency_p95'] = sorted(latencies)[ceil(0.95*cnt)-1]
    log['metrics']['production_accuracy'] /= cnt
    log['metrics']['shadow_accuracy'] /= cnt
    log['metrics']['agreement_rate'] /= cnt
    log['metrics']['accuracy_gain'] = log['metrics']['shadow_accuracy'] - log['metrics']['production_accuracy']

    if log['metrics']['accuracy_gain'] >= criteria['min_accuracy_gain'] and log['metrics']['shadow_latency_p95'] <= criteria['max_latency_p95'] and log['metrics']['agreement_rate'] >= criteria['min_agreement_rate']:
        log['promote'] = True

    return log

