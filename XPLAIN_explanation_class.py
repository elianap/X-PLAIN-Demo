from XPLAIN_utils.LACE_utils.LACE_utils4 import *


class XPLAIN_explanation:
    def __init__(self, explainer, target_class, instance, diff_single, k, error, difference_map):
        self.XPLAIN_explainer_o = explainer
        self.diff_single = diff_single
        self.map_difference = deepcopy(difference_map)
        self.k = k
        self.error = error
        self.instance = instance
        self.target_class = target_class
        self.instance_class_index = explainer.get_class_index(self.target_class)

        c1 = self.XPLAIN_explainer_o.classifier(instance, True)[0]
        self.prob = c1[self.instance_class_index]
