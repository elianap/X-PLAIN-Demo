from collections import Counter

import Orange


def compute_prediction_difference_subset(training_dataset,
                                         instance,
                                         rule_body_indices,
                                         classifier,
                                         instance_class_index,
                                         instance_predictions_cache):
    """
    Compute the prediction difference for an instance in a training_dataset, w.r.t. some
    rules and a class, given a classifier
    """
    rule_attributes = [
        training_dataset.domain.attributes[rule_body_index - 1] for
        rule_body_index in rule_body_indices]

    # Take only the considered attributes from the dataset
    rule_domain = Orange.data.Domain(rule_attributes)
    filtered_dataset = Orange.data.Table().from_table(rule_domain, training_dataset)

    # Count how many times a set of attribute values appears in the dataset
    attribute_sets_occurrences = dict(
        Counter(map(tuple, filtered_dataset.X)).items())

    # For each set of attributes
    differences = [compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                                rule_attributes, rule_domain, training_dataset) for
                   item in
                   attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = classifier(instance, True)[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


def compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                 rule_attributes, rule_domain, training_dataset):
    (attribute_set, occurrences) = item
    perturbed_instance = Orange.data.Instance(training_dataset.domain, instance.list)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_domain[i]] = attribute_set[i]
    # cache_key = tuple(perturbed_instance.x)
    # if cache_key not in instance_predictions_cache:
    #     instance_predictions_cache[cache_key] = classifier(perturbed_instance, True)[0][
    #         instance_class_index]
    # prob = instance_predictions_cache[cache_key]
    prob = classifier(perturbed_instance, True)[0][instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset)
    return difference


# Single explanation. Change 1 value at the time e compute the difference
def compute_prediction_difference_single(instT, classifier, indexI, dataset):
    from copy import deepcopy
    i = deepcopy(instT)
    listaoutput = []

    c1 = classifier(i, True)[0]
    prob = c1[indexI]

    for _ in i.domain.attributes[:]:
        listaoutput.append(0.0)

    t = -1
    for k in dataset.domain.attributes[:]:
        d = Orange.data.Table()
        t = t + 1
        k_a_i = Orange.data.Domain([k])
        filtered_i = d.from_table(k_a_i, dataset)
        c = Counter(map(tuple, filtered_i.X))
        freq = dict(c.items())

        for k_ex in freq:
            inst1 = deepcopy(instT)
            inst1[k] = k_ex[0]
            c1 = classifier(inst1, True)[0]

            prob = c1[indexI]
            test = freq[k_ex] / len(dataset)
            # newvalue=prob*freq[k_ex]/len(dataset)
            newvalue = prob * test
            listaoutput[t] = listaoutput[t] + newvalue

    l = len(listaoutput)

    for i in range(0, l):
        listaoutput[i] = prob - listaoutput[i]
    return listaoutput
