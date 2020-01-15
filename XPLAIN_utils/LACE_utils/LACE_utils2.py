#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def compute_error_approximation(mappa_class, pred, out_data, impo_rules_complete, classname,
                                map_difference):
    PI = pred - mappa_class[classname]
    Sum_Deltas = sum(out_data)
    #UPDATED_EP
    impo_rules_completeC = ", ".join(map(str, list(max(impo_rules_complete, key=len))))

    approx_single_d = abs(PI - Sum_Deltas)
    approx_single_rel = approx_single_d / abs(PI)

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            Delta_impo_rules_completeC = map_difference[impo_rules_completeC.replace(" ", "")]
            PI_approx2 = Delta_impo_rules_completeC
            Sum_Deltas_not_in = 0.0
            # Sum of delta_i for each attribute not included
            for i_out_data in range(0, len(out_data)):
                if str(i_out_data + 1) not in impo_rules_completeC.replace(" ", "").split(","):
                    Sum_Deltas_not_in = Sum_Deltas_not_in + out_data[i_out_data]
        else:
            index = int(impo_rules_completeC.replace(" ", "").split(",")[0]) - 1
            PI_approx2 = out_data[index]
        approx2 = abs(PI - PI_approx2)
        approx_rel2 = approx2 / abs(PI)
    else:
        PI_approx2 = 0.0
        approx_rel2 = 1

    approx2 = abs(PI - PI_approx2)

    return approx_single_rel, approx2, approx_rel2


# MODIFICARE
def computeApproxErrorRule(mappa_class, pred, out_data, impo_rules_complete, classname,
                           map_difference):
    PI = pred - mappa_class[classname]
    impo_rules_completeC = ''.join(str(e) + ", " for e in list(max(impo_rules_complete, key=len)))[
                           :-2]

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            Delta_impo_rules_completeC = map_difference[impo_rules_completeC.replace(" ", "")]
            PI_approx2 = Delta_impo_rules_completeC
        else:
            index = int(impo_rules_completeC.replace(" ", "").split(",")[0]) - 1
            PI_approx2 = out_data[index]
        approx2 = abs(PI - PI_approx2)
        approx_rel2 = approx2 / abs(PI)
    else:
        PI_approx2 = 0.0
        approx2 = 1
        approx_rel2 = 1

    if PI == 0:
        # todo
        approx2 = abs(PI - PI_approx2)

    return approx2, approx_rel2


def getStartKValueSimplified(len_dataset):
    if len_dataset < 150:
        maxN = len_dataset
    elif len_dataset < 1000:
        maxN = int(len_dataset / 2)
    elif len_dataset < 10000:
        maxN = int(len_dataset / 10)
    else:
        maxN = int(len_dataset * 5 / 100)
    return maxN


def computeMappaClass_b(data):
    mappa_class2 = {}
    h = len(data)
    dim_d = len(data[0])
    for d in data[:]:
        c_tmp = d[dim_d - 1].value
        if c_tmp in mappa_class2:
            mappa_class2[c_tmp] = mappa_class2[c_tmp] + 1.0
        else:
            mappa_class2[c_tmp] = 1.0

    for key in mappa_class2.keys():
        mappa_class2[key] = mappa_class2[key] / h

    return mappa_class2


def getUserDefined(args, len_dataset):
    # If parameter Kneighbors is defined, the number of neighbors for the local model is set as user defined. Otherwise the starting value is sqrt(dataset_size)
    if args["Kneighbors"] == None:
        import math
        K_NN = int(round(math.sqrt(len_dataset)))
    else:
        K_NN = int(args["Kneighbors"])

    if args["threshold"] == None:
        threshold = 0.10
    else:
        threshold = float(args["threshold"])
    if args["maxKNN"] == None:
        maxN = getStartKValueSimplified(len_dataset)
    else:
        maxN = int(args["maxKNN"])
    return K_NN, threshold, maxN
