from XPLAIN_utils.LACE_utils.LACE_utils4 import *
from copy import deepcopy

class User_Explanation:
    def __init__(self, explaination,):
        self.lace_explanation = deepcopy(explaination)
        self.instance_explanation = deepcopy(explaination)
        self.id_user_rules=[]


    def updateUserRules(self, explaination, user_rule_id):
        self.instance_explanation=deepcopy(explaination)
        self.id_user_rules.append(user_rule_id)
