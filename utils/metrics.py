import numpy as np


def batch_metric_calculation(llm_answers_batch, y_amb_type_batch, y_amb_keywords_batch):
    metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_keywords':[],
                     'sr':[], 'help_rate': [], 'ambiguity_detection': []}
    for i in range(len(llm_answers_batch)):
        if isinstance(y_amb_keywords_batch[i], str):
            y_amb_keywords = y_amb_keywords_batch[i].split(",")
        metrics = _calculate_metrics(llm_answers_batch[i], 
                                     y_amb_type_batch[i], 
                                     y_amb_keywords)
        for key in metrics:
            metrics_batch[key].append(metrics[key])      
    return metrics_batch

def aggreate(metrics_batch):
    # SR only for preferences (!)
    sr_rates = np.asarray(metrics_batch['sr'])
    sr = np.mean(sr_rates[sr_rates>=0])
    
    amb_det_rates = np.asarray(metrics_batch['ambiguity_detection'])
    amb_detection = np.mean(amb_det_rates[amb_det_rates>=0])

    help_rates = np.asarray(metrics_batch['help_rate'])
    help_rate = np.sum(help_rates[help_rates>=0])/len(help_rates)
    return {'sr_agg': sr, 'amb_detection_agg': amb_detection, 'help_rate_agg': help_rate}
    
                
def _calculate_metrics(llm_answers, y_amb_type, y_amb_keywords):
    return {'llm_answers':llm_answers,
               'y_amb_type': y_amb_type,
               'y_amb_keywords': y_amb_keywords,
                'sr': success_rate(llm_answers, y_amb_keywords), 
               'help_rate': help_rate(llm_answers, y_amb_keywords),
               'ambiguity_detection':ambiguity_detection(llm_answers, y_amb_keywords)}
    
def success_rate(llm_answers, y_amb_keywords):
    if not isinstance(y_amb_keywords, list) or len(y_amb_keywords) == 0:
        return -1
    if not isinstance(llm_answers, list):
        return -1
    sucess_counter = 0
    llm_full_answers = " ".join(llm_answers)
    print(y_amb_keywords)
    for keyword in y_amb_keywords:
        if keyword in llm_full_answers:
            sucess_counter += 1
    return sucess_counter / len(y_amb_keywords)


def ambiguity_detection(llm_answers, y_amb_keywords):
    if not isinstance(y_amb_keywords, list):
        return -1
    if isinstance(llm_answers, list) and len(llm_answers) > 1 and len(y_amb_keywords) > 1:
        return 1
    else:
        return 0

def help_rate(llm_answers, y_amb_keywords):
    if not isinstance(y_amb_keywords, list):
        return -1
    if isinstance(llm_answers, list) and len(llm_answers) > 1:
        return 1
    return 0

def set_size_correctness(data):
    '''
    Set Size Correctness

    measures IoU (intersection over union)
    for predictions and ground truth sets

    Args:
        data (list[tuple[list, list]]): a list of predicted actions and a list of true actions.

    Returns:
        tuple[list[float], dict]: a list of IoU ratios and a dictionary of mean and median IoU ratios.
    '''
    ratios = []

    for pred, truth in data:
        inter = set(pred).intersection(set([truth]))
        union = set(pred).union(set([truth]))
        ratios.append(len(inter) / len(union))
    
    metrics = {}
    metrics['mean'] = np.mean(ratios)
    metrics['median'] = np.median(ratios)

    return ratios, metrics


