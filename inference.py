import numpy as np


def threshold_choose(label_raw, threshold=0.5):
    return 1 * (label_raw >= threshold)


def slice_to_shot(label_array, disrupt_threshold=3, clear_threshold=1):
    disrupt_counter, clear_counter = 0, 0
    for i in range(len(label_array)):
        element = label_array[i]
        if disrupt_counter < disrupt_threshold:
            disrupt_counter += element
            clear_counter += (1 - element)
            if clear_counter >= clear_threshold:
                disrupt_counter = 0
                clear_counter = 0
        else:
            return len(label_array) - i  # Warning time (ms)
    return 0


def evaluation(pre_dict: dict, threshold=0.5, **kwargs):
    tp, fp, tn, fn = 0, 0, 0, 0
    pre_time = dict()
    for shot in pre_dict.keys():
        disruptive = pre_dict[shot][1]
        label = threshold_choose(pre_dict[shot][0], threshold)
        warning_time = slice_to_shot(label, disrupt_threshold=10, **kwargs)
        if disruptive:
            if warning_time >5:
                pre_time[shot] = warning_time
                tp += 1
            else:
                fn += 1
        else:
            if warning_time >5:
                fp += 1
            else:
                tn += 1
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print('tpr:{}'.format(tpr))
    print('fpr:{}'.format(fpr))
    print('average pre_time {}'.format(np.array([*pre_time.values()]).mean()))
    return tpr, fpr, np.array([*pre_time.values()])
