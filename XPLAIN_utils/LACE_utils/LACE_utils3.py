from XPLAIN_utils.LACE_utils.LACE_utils1 import *


# 15-7
def gen_neighbors_info(training_dataset, NearestNeighborsAll, instance, k,
                       unique_filename, classifier, save=True):
    instance_features = instance.x
    nearest_neighbors = NearestNeighborsAll.kneighbors([instance_features], k,
                                                       return_distance=False)

    out_data_raw = []
    lendataset_nearest_neighbors = len(nearest_neighbors[0])
    for i in range(0, lendataset_nearest_neighbors):
        c = classifier(training_dataset[nearest_neighbors[0][i]])
        instanceK = Orange.data.Instance(training_dataset.domain,
                                         training_dataset[
                                             nearest_neighbors[0][i]])
        instanceK.set_class(c[0])
        if i == 0:
            instanceK_i = Orange.data.Instance(training_dataset.domain,
                                               instance)
            c = classifier(instanceK_i)
            instanceTmp = deepcopy(instanceK_i)
            instanceTmp.set_class(c[0])
            out_data_raw.append(instanceTmp)
        out_data_raw.append(instanceK)

    out_data = Orange.data.Table(training_dataset.domain, out_data_raw)

    c = classifier(training_dataset[nearest_neighbors[0][0]])
    instance0 = Orange.data.Instance(training_dataset.domain,
                                     training_dataset[nearest_neighbors[0][0]])
    instance0.set_class(c[0])
    out_data1 = Orange.data.Table(training_dataset.domain, [instance0])

    if save:
        import os
        path = "./" + unique_filename
        if not os.path.exists(path):
            os.makedirs(path)
        toARFF(path + "/Knnres.arff", out_data)
        toARFF(path + "/Filetest.arff", out_data1)
        toARFF(path + "/gen-k0.arff", out_data1)

    return out_data, out_data1


def genNeighborsInfoTraining(training_dataset, NearestNeighborsAll, instanceI,
                             iID, NofKNN, unique_filename, classifier):
    nearest_neighbors = NearestNeighborsAll.kneighbors([instanceI], NofKNN,
                                                       return_distance=False)

    nearest_neighbors_out_data1 = NearestNeighborsAll.kneighbors([instanceI], 1,
                                                                 return_distance=False)

    table = Orange.data.Table
    out_data = Orange.data.Table(training_dataset.domain)
    lendataset_nearest_neighbors = len(nearest_neighbors[0])
    for i in range(0, lendataset_nearest_neighbors):
        #    c=classifier(training_dataset[nearest_neighbors[0][i]])
        instanceK = Orange.data.Instance(training_dataset.domain,
                                         training_dataset[
                                             nearest_neighbors[0][i]])
        #    instanceK.set_class(c[0])
        out_data.append(instanceK)
    c = classifier(training_dataset[nearest_neighbors[0][0]])
    out_data1 = Orange.data.Table(training_dataset.domain)
    instance0 = Orange.data.Instance(training_dataset.domain,
                                     training_dataset[nearest_neighbors[0][0]])
    instance0.set_class(c[0])
    out_data1.append(instance0)

    return out_data, out_data1


def get_relevant_subset_from_local_rules(impo_rules, oldinputAr):
    inputAr = []
    iA = []
    nInputAr = []

    for i2 in range(0, len(impo_rules)):
        intInputAr = []
        val = impo_rules[i2].split(",")
        for i3 in range(0, len(val)):
            intInputAr.append(int(val[i3]))
            iA.append(int(val[i3]))
        nInputAr.append(intInputAr)
    iA2 = list(sorted(set(iA)))
    inputAr.append(iA2)
    if inputAr[0] not in nInputAr:
        nInputAr.append(inputAr[0])
    inputAr = deepcopy(nInputAr)
    oldAr_set = set(map(tuple, oldinputAr))
    # In order to not recompute the prior probability of a Subset again
    newInputAr = [x for x in inputAr if tuple(x) not in oldAr_set]
    oldAr_set = set(map(tuple, oldinputAr))

    return inputAr, nInputAr, newInputAr, oldAr_set
