# Benchmarking notes

## Adults - Naive Bayes

`python -m pyinstrument -r html api.py`

```python
     7                                           def compute_prediction_difference_subset_random_only_existing(training_dataset,
     8                                                                                                         instance,
     9                                                                                                         rule_body_indices,
    10                                                                                                         classifier,
    11                                                                                                         instance_class_index):
    12                                               """
    13                                               Compute the prediction difference for an instance in a training_dataset, w.r..t some
    14                                               rules and a class, given a classifier
    15                                               """
    16        10        361.0     36.1      0.0      print("computePredictionDifferenceSubsetRandomOnlyExisting")
    17        10        889.0     88.9      0.0      print("instance =", instance)
    18        10         98.0      9.8      0.0      print("rule_body_indices =", rule_body_indices)
    19
    20                                               # Dictionary<Sting, Float>
    21                                               # for example: {'3,8,9': 0.9524960582649826, '3,6,8,9': 0.8335597713904336}
    22        10         13.0      1.3      0.0      prediction_difference = 0.0
    23
    24                                               rule_attributes = [
    25        10         22.0      2.2      0.0              training_dataset.domain.attributes[rule_body_index - 1] for
    26        10         43.0      4.3      0.0              rule_body_index in rule_body_indices]
    27
    28                                               # Take only the considered attributes from the dataset
    29        10        994.0     99.4      0.0      rule_domain = Orange.data.Domain(rule_attributes)
    30        10       6793.0    679.3      0.1      filtered_dataset = Orange.data.Table().from_table(rule_domain, training_dataset)
    31
    32                                               # Count how many times a set of attribute values appears in the dataset
    33        10         21.0      2.1      0.0      attribute_sets_occurrences = dict(
    34        10     251341.0  25134.1      4.4          Counter(map(tuple, filtered_dataset.X)).items())
    35
    36        10         31.0      3.1      0.0      print("len(rule_attributes) =", len(rule_attributes),
    37        10        291.0     29.1      0.0            " <=> len(attribute_sets_occurrences) =", len(attribute_sets_occurrences))
    38
    39                                               # For each set of attributes
    40      5376       5882.0      1.1      0.1      for (attribute_set, occurrences) in attribute_sets_occurrences.items():
    41      5366     814103.0    151.7     14.3          prob = compute_prediction_difference_for_class(attribute_set, classifier, instance.list,
    42      5366       5781.0      1.1      0.1                                                         instance_class_index, rule_attributes,
    43      5366    4543855.0    846.8     80.1                                                         rule_domain, training_dataset)
    44                                                   # Update the prediction difference using the weighted average of the
    45                                                   # probability over the frequency of this attribute set in the
    46                                                   # dataset
    47      5366       7027.0      1.3      0.1          prediction_difference += (
    48      5366      30738.0      5.7      0.5                  prob * occurrences / len(training_dataset)
    49                                                   )
    50
    51                                               # p(y=c|x) i.e. Probability that instance x belongs to class c
    52        10       5420.0    542.0      0.1      p = classifier(instance, True)[0][instance_class_index]
    53        10         16.0      1.6      0.0      prediction_differences = p - prediction_difference
    54
    55        10          8.0      0.8      0.0      return prediction_differences
```
