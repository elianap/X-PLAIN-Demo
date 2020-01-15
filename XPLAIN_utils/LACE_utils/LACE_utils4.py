from XPLAIN_utils.LACE_utils.LACE_utils5 import *
from XPLAIN_utils.XPLAIN_utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_context("talk")


def getMinRelevantSet(instT, impo_rules_complete, map_difference):
    minlen = len(instT.domain.attributes)
    minlenname = ""
    minvalue = 0.0
    #UPDATED_PE
    
    impo_rules_completeC=", ".join(map(str, list(max(impo_rules_complete, key=len))))
    #impo_rules_completeC = ''.join(str(e) + ", " for e in list(impo_rules_complete[0]))
    #impo_rules_completeC = impo_rules_completeC[:-2]

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            for k in map_difference:
                if map_difference[k] == map_difference[impo_rules_completeC.replace(" ", "")]:
                    a = k.split(",")
                    if len(a) < minlen:
                        minlen = len(a)
                        minlenname = k
                        minvalue = map_difference[minlenname]
    return minlen, minlenname, minvalue


def plotbarh(x, y, title, label, namefig, colour, min_m, max_m, user_rules=None, save=False):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    y_pos = np.arange(len(x))
    colourU = "#3d3b78"

    if user_rules == None:
        ax.barh(y_pos, y, align='center', color=colour, linewidth='1', edgecolor='black')
    else:
        for y_pos_i in y_pos:
            if x[y_pos_i] in user_rules:
                colorI = colourU
            else:
                colorI = colour
            ax.barh(y_pos_i, y[y_pos_i], align='center', color=colorI, linewidth='1', edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    if min_m != "f" and max_m != "f":
        ax.set_xlim(min_m * 1.2, max_m * 1.1)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(label)
    if save:
        fig.savefig(namefig.split(".png")[0] + ".svg", bbox_inches="tight")
    plt.show()
    plt.close()


def plotbarh_axi(x, y, title, label, namefig, colour, min_m, max_m, ax):
    y_pos = np.arange(len(x))
    ax.barh(y_pos, y, align='center', color=colour, linewidth='1', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    if min_m != "f" and max_m != "f":
        ax.set_xlim(min_m * 1.2, max_m * 1.1)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(label)
    return ax


def getAttrProbSingle(instT, out_data):
    attributes_list = []
    for i in instT.domain.attributes:
        attributes_list.append(str(i.name + "=" + str(instT[i])))
    probSingleNoInt = []
    probSingleNoInt = deepcopy(out_data)
    return attributes_list, probSingleNoInt


def getUnionXY(y_labels, list_single_compare, y_label_mapping, map_difference, impo_rules_completeC, classif_name=False,
               predDiff=False, attributes_list_a=False, minvalue=False):
    list_single_compare_2 = []
    list_single_compare_2 = deepcopy(list_single_compare)
    if impo_rules_completeC not in y_label_mapping.keys():
        if len(impo_rules_completeC.split(",")) != 1:
            list_single_compare_2.append(map_difference[impo_rules_completeC.replace(" ", "")])
            if classif_name != False:
                y_label = "R_" + classif_name + "_U"
            else:
                y_label = "Rule_U"
            y_labels.append(y_label)
            y_label_mapping[str(impo_rules_completeC.replace(" ", ""))] = y_label
            if predDiff != False:
                if map_difference[impo_rules_completeC.replace(" ", "")] != minvalue:
                    predDiff.append(map_difference[impo_rules_completeC.replace(" ", "")])
                    attributes_list_a.append(str(impo_rules_completeC.replace(" ", "")))
        else:
            y_label = "Rule_U"
            y_label_mapping[str(impo_rules_completeC.replace(" ", ""))] = y_label
    return y_labels, list_single_compare_2, y_label_mapping


def getSingles_Rules(attributes_list, probSingleNoInt, impo_rules, map_difference, classif_name=False, predDiff=False,
                     attributes_list_a=False, minvalue=False):
    index_rule = 1
    y_labels = deepcopy(attributes_list)
    y_label_mapping = {}
    list_single_compare = []
    list_single_compare = deepcopy(probSingleNoInt)

    s_imporules = deepcopy(impo_rules)
    s_imporules = sorted(s_imporules, key=len)
    for s_rule in s_imporules:
        if len(s_rule.split(",")) != 1:
            list_single_compare.append(map_difference[s_rule.replace(" ", "")])
            if classif_name != False:
                y_label = "R_" + classif_name + "_" + str(index_rule)
            else:
                y_label = "Rule_" + str(index_rule)
            y_labels.append(y_label)
            y_label_mapping[str(s_rule.replace(" ", ""))] = y_label
            index_rule = index_rule + 1
            if predDiff != False:
                if minlenname != "" and map_difference[s_rule.replace(" ", "")] != minvalue:
                    attributes_list_a.append(str(s_rule.replace(" ", "")))
                    predDiff.append(map_difference[s_rule.replace(" ", "")])
        else:
            y_label = "Rule_" + str(index_rule)
            y_label_mapping[str(s_rule.replace(" ", ""))] = y_label
            index_rule = index_rule + 1
    return list_single_compare, y_labels, y_label_mapping


# TODO: single method _v2 axi
def plotTheInfo_v2(instT, out_data, impo_rules, n_inst, dataname, K, errorFlag, minlenname, minvalue, classname, error,
                   error_single, classif, map_difference, impo_rules_complete, pred_str, save=False):
    classname_f = classname

    if ">" in classname:
        classname_f = classname.replace(">", "gr")
    if "<" in classname:
        classname_f = classname.replace("<", "low")

    import os
    cwd = os.getcwd()
    d_folder = os.path.basename(dataname)
    path = cwd + "/explanations/"
    if save:
        createDir(path)
        path = cwd + "/explanations/" + d_folder + "_" + classif + "_class_" + classname
        createDir(path)

    # Attribute name, Delta_single
    attributes_list, probSingleNoInt = getAttrProbSingle(instT, out_data)
    # attributes_list_a=deepcopy(attributes_list)     #predDiff=deepcopy(probSingleNoInt)

    list_single_compare, y_labels, y_label_mapping = getSingles_Rules(attributes_list, probSingleNoInt, impo_rules,
                                                                      map_difference)

    impo_rules_completeC = ''.join(str(e) + "," for e in list(max(impo_rules_complete, key=len)))[:-1]

    y_labels, list_single_compare_2, y_label_mapping = getUnionXY(y_labels, list_single_compare, y_label_mapping,
                                                                  map_difference, impo_rules_completeC)

    # Unique min-max, in order to have a common scale.
    min_m = min(list_single_compare_2)
    max_m = max(list_single_compare_2)

    plt.style.use('seaborn-talk')
    i_m = ""
    if len(instT.metas) > 0:
        i_m = "x=" + instT.metas[0] + "   "
    title = i_m + "d=" + dataname + " model=" + classif + "\n" + "p(class=" + classname + "|x)=" + pred_str + "   true class= " + str(
        instT.get_class())
    namefig = classif + "_" + str(n_inst) + "_" + classname + '_K' + str(int(K))
    colour = 'lightblue'
    colour = '#bee2e8'

    label2 = "Δ- target class=" + classname

    plotbarh(y_labels, list_single_compare_2, title, label2, path + '/Exp_' + namefig + '_deltas.png', colour, min_m,
             max_m, save=save)
    printMapping_v5(instT, impo_rules, list(map_difference.keys()), y_label_mapping, sep=", ")


def plotTheInfo_v4(instT, out_data, n_inst, dataname, K, classname, classif, map_difference, pred_str, impo_rules,
                   user_rules=[], save=False):
    classname_f = classname

    if ">" in classname:
        classname_f = classname.replace(">", "gr")
    if "<" in classname:
        classname_f = classname.replace("<", "low")

    import os
    cwd = os.getcwd()
    d_folder = os.path.basename(dataname)
    path = cwd + "/explanations/"
    if save:
        createDir(path)
        path = cwd + "/explanations/" + d_folder + "_" + classif + "_class_" + classname
        createDir(path)

    attributes_list, probSingleNoInt = getAttrProbSingle(instT, out_data)
    attributes_list_a = deepcopy(attributes_list)
    predDiff = deepcopy(probSingleNoInt)

    y_labels = deepcopy(attributes_list)
    predDiffComplete = deepcopy(probSingleNoInt)
    user_str = [', '.join(map(str, i)) for i in user_rules]

    y_labels, predDiffComplete, y_label_mapping, index_rule = getXYExplanationRules(y_labels, predDiffComplete,
                                                                                    map_difference, rules=list(
            set(impo_rules + user_str)), uRules=user_rules)

    r1 = list(set([i.replace(",", ", ") for i in list(map_difference.keys())]))
    r2 = list(set(impo_rules + user_str))
    r_u_all = list(set(r1 + r2))
    if user_rules != []:
        printMapping_v5(instT, r_u_all, list(map_difference.keys()), y_label_mapping, sep=", ")
        user_rules = [y_label_mapping[','.join(map(str, i))] for i in user_rules]
    else:
        printMapping_v5(instT, impo_rules, list(map_difference.keys()), y_label_mapping, sep=", ")

    # Unique min-max, in order to have a common scale.
    min_m = min(predDiffComplete)
    max_m = max(predDiffComplete)

    plt.style.use('seaborn-talk')
    i_m = ""
    if len(instT.metas) > 0:
        i_m = "x=" + instT.metas[0] + "   "
    title = i_m + "d=" + dataname + " model=" + classif + "\n" + "p(class=" + classname + "|x)=" + pred_str + "   true class= " + str(
        instT.get_class())
    namefig = classif + "_" + str(n_inst) + "_" + classname + '_K' + str(int(K))
    colour = 'lightblue'
    colour = '#bee2e8'

    label2 = "Δ target class=" + classname

    plotbarh(y_labels, predDiffComplete, title, label2, path + '/Expl_' + namefig + '.png', colour, min_m, max_m,
             user_rules, save=save)


def plotTheInfo_axi(instT, out_data, impo_rules, n_inst, dataname, K, errorFlag, minlenname, minvalue, classname, error,
                    error_single, classif, map_difference, impo_rules_complete, pred_str, axi, save=False):
    classname_f = classname

    if ">" in classname:
        classname_f = classname.replace(">", "gr")
    if "<" in classname:
        classname_f = classname.replace("<", "low")

    import os
    cwd = os.getcwd()
    d_folder = os.path.basename(dataname)
    path = cwd + "/explanations/"
    if save:
        createDir(path)
        path = cwd + "/explanations/" + d_folder + "_" + classif + "_class_" + classname
        createDir(path)

    # Attribute name, Delta_single
    attributes_list, probSingleNoInt = getAttrProbSingle(instT, out_data)
    # attributes_list_a=deepcopy(attributes_list)     #predDiff=deepcopy(probSingleNoInt)

    list_single_compare, y_labels, y_label_mapping = getSingles_Rules(attributes_list, probSingleNoInt, impo_rules,
                                                                      map_difference)

    impo_rules_completeC = ''.join(str(e) + "," for e in list(max(impo_rules_complete, key=len)))[:-1]

    y_labels, list_single_compare_2, y_label_mapping = getUnionXY(y_labels, list_single_compare, y_label_mapping,
                                                                  map_difference, impo_rules_completeC)

    # Unique min-max, in order to have a common scale.
    min_m = min(list_single_compare_2)
    max_m = max(list_single_compare_2)

    plt.style.use('seaborn-talk')
    i_m = ""
    if len(instT.metas) > 0:
        i_m = "x=" + instT.metas[0] + "   "
    title = i_m + "d=" + dataname + " model=" + classif + "\n" + "p(class=" + classname + "|x)=" + pred_str + "   true class= " + str(
        instT.get_class())
    namefig = classif + "_" + str(n_inst) + "_" + classname + '_K' + str(int(K))
    colour = 'lightblue'
    colour = '#bee2e8'

    label2 = "Δ target class=" + classname

    plotbarh_axi(y_labels, list_single_compare_2, title, label2, path + '/SRC_' + namefig + '.png', colour, min_m,
                 max_m, axi)
    printMapping_v5(instT, impo_rules, list(map_difference.keys()), y_label_mapping, sep=", ")


def getXYExplanationRules(y_labels, predDiffComplete, map_difference, rules=[], uRules=False):
    index_rule = 1
    index_rule_user = 1
    y_label_mapping = {}
    if uRules:
        uRules = [','.join(map(str, i)) for i in uRules]
    rules = list(set([i.replace(" ", "") for i in rules] + list(map_difference.keys())))
    for k in sorted(rules, key=len):
        if uRules != False and k in uRules:
            y_label = "Rule_I_" + str(index_rule_user)
            index_rule_user = index_rule_user + 1
        else:
            y_label = "Rule_" + str(index_rule)
            index_rule = index_rule + 1
        if len(k.split(",")) != 1:
            predDiffComplete.append(map_difference[k])
            y_labels.append(y_label)
        y_label_mapping[k] = y_label

    return y_labels, predDiffComplete, y_label_mapping, index_rule


def getSingles_Rules(attributes_list, probSingleNoInt, impo_rules, map_difference, classif_name=False, predDiff=False,
                     attributes_list_a=False, minvalue=False):
    index_rule = 1
    y_labels = deepcopy(attributes_list)
    y_label_mapping = {}
    list_single_compare = []
    list_single_compare = deepcopy(probSingleNoInt)

    s_imporules = deepcopy(impo_rules)
    s_imporules = sorted(s_imporules, key=len)
    for s_rule in s_imporules:
        if len(s_rule.split(",")) != 1:
            list_single_compare.append(map_difference[s_rule.replace(" ", "")])
            if classif_name != False:
                y_label = "R_" + classif_name + "_" + str(index_rule)
            else:
                y_label = "Rule_" + str(index_rule)
            y_labels.append(y_label)
            y_label_mapping[str(s_rule.replace(" ", ""))] = y_label
            index_rule = index_rule + 1
            if predDiff != False:
                if minlenname != "" and map_difference[s_rule.replace(" ", "")] != minvalue:
                    attributes_list_a.append(str(s_rule.replace(" ", "")))
                    predDiff.append(map_difference[s_rule.replace(" ", "")])
        else:
            y_label = "Rule_" + str(index_rule)
            y_label_mapping[str(s_rule.replace(" ", ""))] = y_label
            index_rule = index_rule + 1
    return list_single_compare, y_labels, y_label_mapping
