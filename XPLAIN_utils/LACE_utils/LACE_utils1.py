import os
import pickle
from copy import deepcopy

import Orange

MAX_SAMPLE_COUNT = 100


def import_dataset_N_evaluations(dataname, n_insts, randomic,
                                 datasetExplain=False):
    if datasetExplain:
        return import_datasets(dataname, n_insts, randomic)
    else:
        return import_dataset(dataname, n_insts, randomic)


def import_dataset(dataset_name, explain_indices, random_explain_dataset):
    if dataset_name[-4:] == "arff":
        print(dataset_name)
        dataset = loadARFF(dataset_name)
    else:
        dataset = Orange.data.Table(dataset_name)        
        #TD 
        if False in [i.is_discrete for i in dataset[0].domain.attributes]:
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EqualFreq(3)
            dataset = disc(dataset)
            toARFF(dataset_name.split(".")[0]+".arff", dataset)
            dataset = loadARFF(dataset_name.split(".")[0]+".arff")

    dataset_len = len(dataset)
    training_indices = list(range(dataset_len))

    if random_explain_dataset:
        import random
        random.seed(1)

        # small dataset
        if dataset_len < (2 * MAX_SAMPLE_COUNT):
            samples = int(0.2 * dataset_len)
        else:
            samples = MAX_SAMPLE_COUNT

        # Randomly pick some instances to remove from the training dataset and use in the
        # explain dataset
        explain_indices = list(random.sample(training_indices, samples))
        #explain_indices.sort()
    for i in explain_indices:
        training_indices.remove(i)

    training_dataset = Orange.data.Table.from_table_rows(dataset, training_indices)
    explain_dataset = Orange.data.Table.from_table_rows(dataset, explain_indices)

    return training_dataset, explain_dataset, len(training_dataset), \
           [str(i) for i in explain_indices]


def import_datasets(dataname, n_insts, randomic):
    if dataname[-4:] == "arff":
        dataset = loadARFF(dataname)
        dataname_to_explain = dataname[:-5] + "_explain.arff"
        dataset_to_explain = loadARFF(dataname_to_explain)
    else:
        dataset = Orange.data.Table(dataname)
        dataname_to_explain = dataname[:-5] + "_explain"
        dataset_to_explain = Orange.data.Table(dataname_to_explain)
    len_dataset = len(dataset)

    len_dataset_to_explain = len(dataset_to_explain)

    if randomic:
        import random
        # 7
        random.seed(7)
        n_insts = list(random.sample(range(len_dataset_to_explain), 300))
        n_insts = [str(i) for i in n_insts]

    n_insts_int = list(map(int, n_insts))

    explain_dataset = Orange.data.Table.from_table_rows(dataset_to_explain,
                                                        n_insts_int)

    training_dataset = deepcopy(dataset)
    return training_dataset, explain_dataset, len_dataset, n_insts


def toARFF(filename, table, try_numericize=0):
    """Save class:`Orange.data.Table` to file in Weka's ARFF format"""
    t = table
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    # print( filename
    f = open(filename + '.arff', 'w')
    f.write('@relation %s\n' % t.domain.class_var.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.class_var)
    for i in ats:
        real = 1
        if i.is_discrete == 1:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = float(j)
                    except:
                        real = 0  # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if iname.find(" ") != -1:
            iname = "'%s'" % iname
        if real == 1:
            f.write('@attribute %s real\n' % iname)
        else:
            f.write('@attribute %s { ' % iname)
            x = []
            for j in i.values:
                s = str(j)
                if s.find(" ") == -1:
                    x.append("%s" % s)
                else:
                    x.append("'%s'" % s)
            for j in x[:-1]:
                f.write('%s,' % j)
            f.write('%s }\n' % x[-1])
    f.write('@data\n')
    for j in t:
        x = []
        for i in range(len(ats)):
            s = str(j[i])
            if s.find(" ") == -1:
                x.append("%s" % s)
            else:
                x.append("'%s'" % s)
        for i in x[:-1]:
            f.write('%s,' % i)
        f.write('%s\n' % x[-1])
    f.close()


def loadARFF_Weka(filename):
    if not os.path.exists(filename) and os.path.exists(filename + ".arff"):
        filename = filename + ".arff"
    with open(filename, 'r') as f:

        attributes = []
        name = ''
        in_header = False  # header
        rows = []

        for line in f.readlines():
            line = line.rstrip("\n\r")  # strip trailing whitespace
            line = line.replace('\t', ' ')  # get rid of tabs
            line = line.split('%')[0]  # strip comments
            if len(line.strip()) == 0:  # ignore empty lines
                continue
            if not in_header and line[0] != '@':
                print(("ARFF import ignoring:", line))
            if in_header:  # Header
                if line[0] == '{':  # sparse data format, begin with '{', ends with '}'
                    r = [None] * len(attributes)
                    row = line[1:-1]
                    row = row.split(',')
                    for xs in row:
                        y = xs.split(" ")
                        if len(y) != 2:
                            raise ValueError("the format of the data is error")
                        r[int(y[0])] = y[1]
                    rows.append(r)
                else:  # normal data format, split by ','
                    row = line.split(',')
                    r = []
                    for xs in row:
                        y = xs.strip(" ")
                        if len(y) > 0:
                            if y[0] == "'" or y[0] == '"':
                                r.append(xs.strip("'\""))
                            else:
                                ns = xs.split()
                                for ls in ns:
                                    if len(ls) > 0:
                                        r.append(ls)
                        else:
                            r.append('?')
                    rows.append(r[:len(attributes)])
            else:  # Data
                y = []
                for cy in line.split(' '):
                    if len(cy) > 0:
                        y.append(cy)
                if str.lower(y[0][1:]) == 'data':
                    in_header = True
                elif str.lower(y[0][1:]) == 'relation':
                    name = str.strip(y[1])
                elif str.lower(y[0][1:]) == 'attribute':
                    if y[1][0] == "'":
                        atn = y[1].strip("' ")
                        idx = 1
                        while y[idx][-1] != "'":
                            idx += 1
                            atn += ' ' + y[idx]
                        atn = atn.strip("' ")
                    else:
                        atn = y[1]
                    z = line.split('{')
                    w = z[-1].split('}')
                    if len(z) > 1 and len(w) > 1:
                        # there is a list of values
                        vals = []
                        for y in w[0].split(','):
                            sy = y.strip(" '\"")
                            if len(sy) > 0:
                                vals.append(sy)
                        a = Orange.data.DiscreteVariable.make(atn, vals, True, 0)
                    else:
                        a = Orange.data.variable.ContinuousVariable.make(atn)
                    attributes.append(a)

        # generate the domain
        if attributes[-1].name == name:
            domain = Orange.data.Domain(attributes[:-1], attributes[-1])
        else:
            new_attr = []
            for att in attributes:
                if att != name:
                    new_attr.append(att)
            domain = Orange.data.Domain(new_attr)

        instances = [Orange.data.Instance(domain, row) for row in rows]

        table = Orange.data.Table.from_list(domain, instances)
        table.name = name

        return table


def loadARFF(filename, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Weka ARFF format
       if there exists no .xml file with the same name. If it does, a multi-label
       dataset is read and returned.
    """
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml"
        arff_name = filename + ".arff"
        return Orange.multilabel.mulan.trans_mulan_data(xml_name, arff_name)
    else:
        return loadARFF_Weka(filename)


def printTree(classifier, name):
    features_names = get_features_names(classifier)
    from io import StringIO
    import pydotplus
    dot_data = StringIO()
    from sklearn import tree
    if features_names != None:
        tree.export_graphviz(classifier.skl_model, out_file=dot_data,
                             feature_names=features_names, filled=True,
                             rounded=True, special_characters=True)
    else:
        tree.export_graphviz(classifier.skl_model, out_file=dot_data,
                             filled=True, rounded=True,
                             special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(name + "_tree.pdf")


def get_features_names(classifier):
    features_names = []
    for i in range(0, len(classifier.domain.attributes)):
        if ">" in classifier.domain.attributes[i].name:
            features_names.append(
                classifier.domain.attributes[i].name.replace(">", "gr"))

        elif "<" in classifier.domain.attributes[i].name:
            features_names.append(
                classifier.domain.attributes[i].name.replace("<", "low"))
        else:
            features_names.append(classifier.domain.attributes[i].name)

    return features_names


def getClassifier(training_dataset, args, exit):
    classif = args["classifier"]
    classifier = None
    reason = args
    if classif == "tree":
        if (args["classifierparameter"] == None):
            measure = "gini"
        else:
            measure = args["classifierparameter"].split("-")[0]
        if (measure) != "gini" and (measure) != "entropy":
            measure = "entropy"
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnertree = Orange.classification.SklTreeLearner(
            preprocessors=continuizer, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=1)
        # learnertree=Orange.classification.SklTreeLearner(preprocessors=continuizer, random_state=1)

        classifier = learnertree(training_dataset)

        printTree(classifier, training_dataset.name)



    elif classif == "nb":
        learnernb = Orange.classification.NaiveBayesLearner()
        classifier = learnernb(training_dataset)

    elif classif == "nn":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnernet = Orange.classification.NNClassificationLearner(
            preprocessors=continuizer, random_state=42,
            max_iter=100)
        print(learnernet)

        classifier = learnernet(training_dataset)


    elif classif == "rf":
        import random
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.RandomForestLearner(
            preprocessors=continuizer, random_state=42)
        classifier = learnerrf(training_dataset)

    elif classif == "svm":
        import random
        learnerrf = Orange.classification.SVMLearner(preprocessors=continuizer)
        classifier = learnerrf(training_dataset)

    elif classif == "knn":
        if args["classifierparameter"] == None:
            exit = 1
            reason = "k - missing the K parameter"
        elif (len(args["classifierparameter"].split("-")) == 1):
            KofKNN = int(args["classifierparameter"].split("-")[0])
            distance = ""
        else:
            KofKNN = int(args["classifierparameter"].split("-")[0])
            distance = args["classifierparameter"].split("-")[1]
        if exit != 1:
            if distance == "eu":
                metricKNN = 'euclidean'
            elif distance == "ham":
                metricKNN = 'hamming'
            elif distance == "man":
                metricKNN = 'manhattan'
            elif distance == "max":
                metricKNN = 'maximal'
            else:
                metricKNN = 'euclidean'
            continuizer = Orange.preprocess.Continuize()
            continuizer.multinomial_treatment = continuizer.Indicators
            knnLearner = Orange.classification.KNNLearner(
                preprocessors=continuizer, n_neighbors=KofKNN,
                metric=metricKNN, weights='uniform', algorithm='auto',
                metric_params=None)
            classifier = knnLearner(training_dataset)
    else:
        reason = "Classification model not available"
        exit = 1

    return classifier, exit, reason


def useExistingModel(args):
    import os
    if os.path.exists("./models") == False:
        os.makedirs("./models")
    m = ""
    if args["classifierparameter"] != None:
        m = "-" + args["classifierparameter"]
    file_path = "./models/" + args["dataset"] + "-" + args["classifier"] + m
    if (os.path.exists(file_path) == True):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        modelname = ""
        if args["classifier"] == "tree":
            modelname = "<class 'Orange.classification.tree.SklTreeClassifier'>"
        elif args["classifier"] == "nb":
            modelname = "<class 'Orange.classification.naive_bayes.NaiveBayesModel'>"
        elif args["classifier"] == "nn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif args["classifier"] == "knn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif args["classifier"] == "rf":
            modelname = "<class 'Orange.classification.random_forest.RandomForestClassifier'>"
        else:
            return False

        if str(type(model)) == modelname:
            return model

    return False


def useExistingModel_v2(classif, classifierparameter, dataname):
    import os
    if os.path.exists("./models") == False:
        os.makedirs("./models")
    m = ""
    if classifierparameter != None:
        m = "-" + classifierparameter
    file_path = "./models/" + dataname + "-" + classifierparameter + m
    if (os.path.exists(file_path) == True):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        modelname = ""
        if classif == "tree":
            modelname = "<class 'Orange.classification.tree.SklTreeClassifier'>"
        elif classif == "nb":
            modelname = "<class 'Orange.classification.naive_bayes.NaiveBayesModel'>"
        elif classif == "nn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif classif == "knn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif classif == "rf":
            modelname = "<class 'Orange.classification.random_forest.RandomForestClassifier'>"
        else:
            return False

        if str(type(model)) == modelname:
            return model

    return False


def getClassifier_v2(training_dataset, classif, classifierparameter, exit):
    classif = classif
    classifier = None
    reason = ""
    if classif == "tree":
        if (classifierparameter == None):
            measure = "gini"
        else:
            measure = classifierparameter.split("-")[0]
        if (measure) != "gini" and (measure) != "entropy":
            measure = "entropy"
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnertree = Orange.classification.SklTreeLearner(
            preprocessors=continuizer, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=1)
        # learnertree=Orange.classification.SklTreeLearner(preprocessors=continuizer, random_state=1)

        classifier = learnertree(training_dataset)

        printTree(classifier, training_dataset.name)
    elif classif == "nb":
        learnernb = Orange.classification.NaiveBayesLearner()
        classifier = learnernb(training_dataset)
    elif classif == "nn":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnernet = Orange.classification.NNClassificationLearner(
            preprocessors=continuizer, random_state=42,
            max_iter=1000)

        classifier = learnernet(training_dataset)
    elif classif == "rf":
        import random
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.RandomForestLearner(
            preprocessors=continuizer, random_state=42)
        classifier = learnerrf(training_dataset)
    elif classif == "svm":
        import random
        learnerrf = Orange.classification.SVMLearner(preprocessors=continuizer)
        classifier = learnerrf(training_dataset)
    elif classif == "knn":
        if classifierparameter == None:
            KofKNN=1
            distance="eu"
            #exit = 1
            #reason = "k - missing the K parameter"
        elif (len(classifierparameter.split("-")) == 1):
            KofKNN = int(classifierparameter.split("-")[0])
            distance = ""
        else:
            KofKNN = int(classifierparameter.split("-")[0])
            distance = classifierparameter.split("-")[1]
        if exit != 1:
            if distance == "eu":
                metricKNN = 'euclidean'
            elif distance == "ham":
                metricKNN = 'hamming'
            elif distance == "man":
                metricKNN = 'manhattan'
            elif distance == "max":
                metricKNN = 'maximal'
            else:
                metricKNN = 'euclidean'
            continuizer = Orange.preprocess.Continuize()
            continuizer.multinomial_treatment = continuizer.Indicators
            knnLearner = Orange.classification.KNNLearner(
                preprocessors=continuizer, n_neighbors=KofKNN,
                metric=metricKNN, weights='uniform', algorithm='auto',
                metric_params=None)
            classifier = knnLearner(training_dataset)
    else:
        reason = "Classification model not available"
        exit = 1

    return classifier, exit, reason


def createDir(outdir):
    try:
        os.makedirs(outdir)
    except:
        pass
