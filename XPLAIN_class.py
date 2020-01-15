# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import subprocess

# noinspection PyUnresolvedReferences
import sklearn.neighbors

from XPLAIN_explanation_class import XPLAIN_explanation
# noinspection PyUnresolvedReferences
from XPLAIN_utils.LACE_utils.LACE_utils2 import getStartKValueSimplified, \
    computeMappaClass_b, compute_error_approximation
# noinspection PyUnresolvedReferences
from XPLAIN_utils.LACE_utils.LACE_utils3 import gen_neighbors_info, \
    get_relevant_subset_from_local_rules, getClassifier_v2
from XPLAIN_utils.LACE_utils.LACE_utils4 import *
from XPLAIN_utils.global_explanation import *

ERROR_DIFFERENCE_THRESHOLD = 0.01
TEMPORARY_FOLDER_NAME = "tmp"
ERROR_THRESHOLD = 0.02


class XPLAIN_explainer:
    def __init__(self, dataset_name, classifier_name, classifier_parameter=None,
                 KneighborsUser=None, maxKNNUser=None, threshold_error=None,
                 use_existing_model=False, save_model=False,
                 random_explain_dataset=False):

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.present = False

        # Temporary folder
        import uuid
        self.unique_filename = os.path.join(TEMPORARY_FOLDER_NAME,
                                            str(uuid.uuid4()))
        self.datanamepred = "./" + self.unique_filename + "/gen-k0.arff"
        should_exit = 0

        # The adult and compas dataset are already splitted in training and explain set.
        # The training set is balanced.
        self.explain_indices = []

        explain_dataset_indices = []
        if dataset_name == "datasets/adult_d.arff" \
                or dataset_name == "datasets/compas-scores-two-years_d.arff":
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = \
                import_datasets(
                    dataset_name, explain_dataset_indices, random_explain_dataset)
        else:
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = \
                import_dataset(
                    dataset_name, explain_dataset_indices, random_explain_dataset)

        self.K, _, self.max_K = get_KNN_threshold_max(KneighborsUser,
                                                      self.training_dataset_len,
                                                      threshold_error,
                                                      maxKNNUser)

        # If the user specifies to use an existing model, the model is used (if available).
        # Otherwise it is trained.
        if use_existing_model:
            # "Check if the model exist...
            self.classifier = useExistingModel_v2(classifier_name,
                                                  classifier_parameter,
                                                  dataset_name)
            if self.classifier:
                self.present = True
                # The model exists, we'll use it
            # The model does not exist, we'll train it")
        if use_existing_model is None or self.present == False:
            self.classifier, should_exit, reason = getClassifier_v2(
                self.training_dataset, classifier_name, classifier_parameter,
                should_exit)

        if should_exit == 1:
            exit(-1)

        # Save the model only if required and it is not already saved.
        if save_model:
            # "Saving the model..."
            m = ""
            if classifier_parameter is not None:
                m = "-" + classifier_parameter
            createDir("./models")
            with open("./models/" + dataset_name + "-" + classifier_name + m,
                      "wb") as f:
                pickle.dump(self.classifier, f)

        self.map_names_class = {}
        num_i = 0
        for i in self.training_dataset.domain.class_var.values:
            self.map_names_class[num_i] = i
            num_i += 1
        self.labels = list(self.map_names_class.keys())

        self.dataset_name = dataset_name.split("/")[-1]

        self.NofClass = len(self.training_dataset.domain.class_var.values[:])

        # Compute the neighbors of the instanceId
        metric_knna = 'euclidean'
        self.NearestNeighborsAll = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.training_dataset), metric=metric_knna,
            algorithm='auto', metric_params=None).fit(self.training_dataset.X)
        self.mappa_single = {}

        self.firstInstance = 1

        self.starting_K = self.K

        self.mappa_class = computeMappaClass_b(self.training_dataset)
        self.count_inst = -1
        self.mispredictedInstances = None
        self.classes = list(self.map_names_class.values())

    def get_class_index(self, class_name):
        class_index = -1
        for i in self.training_dataset.domain.class_var.values:
            class_index += 1
            if i == class_name:
                return class_index

    def getMispredicted(self, mispred_class=False):
        self.mispredictedInstances = []
        count_inst = 0
        for n_ist in self.explain_indices:
            instanceI = Orange.data.Instance(self.explain_dataset.domain,
                                             self.explain_dataset[count_inst])
            c = self.classifier(instanceI, False)
            if instanceI.get_class() != self.map_names_class[c[0]]:
                if mispred_class != False:
                    if instanceI.get_class() == mispred_class:
                        self.mispredictedInstances.append(n_ist)
                else:
                    self.mispredictedInstances.append(n_ist)
            count_inst = count_inst + 1
        return self.mispredictedInstances

    def interactiveTargetClassComparison(self, instID):
        from ipywidgets import HBox, VBox
        classes = ["predicted", "trueLabel"] + self.classes[:]
        w1 = widgets.Dropdown(
            options=classes,
            description='1º',
            value="predicted",
            disabled=False
        )
        w2 = widgets.Dropdown(
            options=classes,
            description='2º',
            value="trueLabel",
            disabled=False
        )
        hClasses = VBox([w1, w2])
        l = widgets.Label(value='Select target classes:')
        display(l)
        display(hClasses)

        def clearAndShow(btNewObj):
            clear_output()
            display(l)
            display(hClasses)
            display(h)

        def getExplainInteractiveButton(btn_object):
            e1, e2 = self.getExplanationComparison(instID, w1.value, w2.value)

        btnTargetC = widgets.Button(description='Compute')
        btnTargetC.on_click(getExplainInteractiveButton)
        btnNewSel = widgets.Button(description='Clear')
        btnNewSel.on_click(clearAndShow)
        h = HBox([btnTargetC, btnNewSel])
        display(h)

    def getMispredictedTrueLabelComparison(self, instID):
        e1, e2 = self.getExplanationComparison(instID, "predicted", "trueLabel")

    def getExplanationComparison(self, Sn_inst, targetClass1,
                                 targetClass2=None):

        if targetClass1 == targetClass2:
            print("Same target class")
            return self.explain_instance(Sn_inst, targetClass1), None

        if targetClass1 == "predicted" and targetClass2 == None:
            print("Predicted class")
            return self.explain_instance(Sn_inst), None

        predicted, true = self.getPredictedandTrueClassById(Sn_inst)

        if targetClass1 == None:
            targetClass1 = "predicted"
        if targetClass2 == None:
            targetClass2 = "predicted"

        if targetClass1 == "predicted" or targetClass2 == "predicted":
            if predicted == targetClass1 or predicted == targetClass2:
                print("Predicted class = user target class ")
                return self.explain_instance(Sn_inst), None
            if targetClass1 == "trueLabel" or targetClass2 == "trueLabel":
                if true == predicted:
                    print("True class = predicted class ")
                    return self.explain_instance(Sn_inst), None
        if targetClass1 == "trueLabel" or targetClass2 == "trueLabel":
            if true == targetClass1 or true == targetClass2:
                print("True class = user target class ")
                return self.explain_instance(Sn_inst), None

        fig2 = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig2.add_subplot(1, 2, 1)
        explanation_1, ax1 = self.getExplanation_i_axis(ax1, Sn_inst,
                                                        targetClass1)
        ax2 = fig2.add_subplot(1, 2, 2)
        explanation_2, ax2 = self.getExplanation_i_axis(ax2, Sn_inst,
                                                        targetClass2)
        plt.tight_layout()
        plt.show()
        return explanation_1, explanation_2

    def getInstanceById(self, Sn_inst):
        count_inst = self.explain_indices.index(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        return instTmp2

    def getPredictedandTrueClassById(self, Sn_inst):
        i = self.getInstanceById(Sn_inst)
        c = self.classifier(i, False)
        return self.map_names_class[c[0]], str(i.get_class())

    def getPredictedandTrueClassByInstance(self, i):
        c = self.classifier(i, False)
        return self.map_names_class[c[0]], str(i.get_class())

    def explain_instance(self, instance, target_class):

        c = self.classifier(instance, False)
        target_class_index = self.get_class_index(target_class)

        self.starting_K = self.K
        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.training_dataset_len), self.starting_K)

        # Initialize k and error to be defined in case the for loop is not entered
        k = self.starting_K
        old_error = 10.0
        error = 1e9
        single_attribute_differences = {}
        pred = 0.0
        difference_map = {}

        first_iteration = True

        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        cached_subset_differences = {}
        instance_predictions_cache = {}

        all_rule_body_indices = []

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            # Compute the prediction difference of single attributes only on the
            # first iteration
            if first_iteration:
                pred = self.classifier(instance, True)[0][target_class_index]
                single_attribute_differences = compute_prediction_difference_single(instance,
                                                                                    self.classifier,
                                                                                    target_class_index,
                                                                                    self.training_dataset)

            PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences = self.compute_lace_step(
                cached_subset_differences, instance,
                instance_predictions_cache,
                k, all_rule_body_indices, target_class, target_class_index, pred,
                single_attribute_differences)

            # If we have reached the minimum or we are stuck in a local minimum
            if (error < ERROR_THRESHOLD) or ((abs(error) - abs(
                    old_error)) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration):
                break
            else:
                first_iteration = False
                old_error = error
        instance_explanation = XPLAIN_explanation(self,
                                                  target_class,
                                                  instance,
                                                  single_attribute_differences,
                                                  k,
                                                  error,
                                                  difference_map)
        # Remove the temporary folder and dir
        import shutil
        if os.path.exists("./" + self.unique_filename):
            shutil.rmtree("./" + self.unique_filename)

        return instance_explanation

    def compute_lace_step(self, cached_subset_differences, instance,
                          instance_predictions_cache, k, old_input_ar, target_class,
                          target_class_index, pred, single_attribute_differences):
        print(f"compute_lace_step k={k}")

        gen_neighbors_info(self.training_dataset, self.NearestNeighborsAll, instance, k,
                           self.unique_filename, self.classifier)
        subprocess.call(['java', '-jar', 'AL3.jar', '-no-cv', '-t',
                         ('./' + self.unique_filename + '/Knnres.arff'), '-T',
                         ('./' + self.unique_filename + '/Filetest.arff'),
                         '-S', '1.0', '-C', '50.0', '-PN',
                         ("./" + self.unique_filename), '-SP', '10', '-NRUL',
                         '1'], stdout=subprocess.DEVNULL)
        with open("./" + self.unique_filename + "/impo_rules.txt",
                  "r") as myfile:
            importance_rules_lines = myfile.read().splitlines()
            # Remove rules which contain all attributes: we are not interested in a rule composed of
            # all the attributes values. By definition, its relevance is prob(y=c)-prob(c)
            importance_rules_lines = [rule_str for rule_str in importance_rules_lines if
                                      len(rule_str.split(",")) != len(instance.domain.attributes)]

        rule_bodies_indices, n_input_ar, new_input_ar, old_ar_set = \
            get_relevant_subset_from_local_rules(
                importance_rules_lines, old_input_ar)
        impo_rules_complete = deepcopy(rule_bodies_indices)

        # Cache the subset calculation for repeated rule subsets.
        difference_map = {}
        for rule_body_indices in rule_bodies_indices:
            # Consider only rules with more than 1 attribute since we compute the differences
            # for single attribute changes already in compute_prediction_difference_single
            if len(rule_body_indices) == 1:
                #Update Eliana - To output also rule of one element
                difference_map[str(rule_body_indices[0])] = single_attribute_differences[rule_body_indices[0]-1]
                continue
            if len(rule_body_indices) < 1:
                continue

            subset_difference_cache_key = tuple(rule_body_indices)
            if subset_difference_cache_key not in cached_subset_differences:
                cached_subset_differences[
                    subset_difference_cache_key] = compute_prediction_difference_subset(
                    self.training_dataset, instance, rule_body_indices,
                    self.classifier, target_class_index, instance_predictions_cache)

            difference_map_key = ",".join(map(str, rule_body_indices))
            difference_map[difference_map_key] = cached_subset_differences[
                subset_difference_cache_key]

        error_single, error, PI_rel2 = compute_error_approximation(self.mappa_class,
                                                                   pred,
                                                                   single_attribute_differences,
                                                                   impo_rules_complete,
                                                                   target_class,
                                                                   difference_map)
        old_input_ar += rule_bodies_indices

        return PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences

    def visualizePoints(self, datapoints, Sn_inst=None, reductionMethod="mca"):
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn import decomposition
        if Sn_inst != None:
            count_inst = self.explain_indices.index(Sn_inst)
            n_inst = int(Sn_inst)
            instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                            self.explain_dataset[count_inst])
            c = self.classifier(instTmp2, False)
            labelledInstance = deepcopy(instTmp2)

        X = datapoints.X
        y = datapoints.Y

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            if Sn_inst != None:
                istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(datapoints)):
                dataK.append(datapoints[k].list)

            columnsA = [i.name for i in datapoints.domain.variables]

            if datapoints.domain.metas != ():
                for i in range(0, len(datapoints.domain.metas)):
                    columnsA.append(datapoints.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in datapoints.domain.attributes]
            Xa = data[columnsA]
            y = datapoints.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True, engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            if Sn_inst != None:
                istance_transformed = mca.transform([[labelledInstance[i].value
                                                      for i in
                                                      labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            if Sn_inst != None:
                XX = np.vstack([X, labelledInstance.x])
                label_istance = float(
                    max(list(self.map_names_class.keys())) + 1)
                yy = np.concatenate((y, np.array([label_istance])))
            else:
                XX = X
                yy = y
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)

        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)

        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        if Sn_inst != None:
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            if Sn_inst != None:
                XX = np.vstack([X, istance_transformed])
                yy = np.concatenate((y, np.array([label_istance])))
            else:
                XX = X
                yy = y
            fig = plt.figure(figsize=(5.5, 3))
            ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            if Sn_inst != None:
                label_values.append(int(label_istance))
        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))

        colors = [sc.cmap(sc.norm(i)) for i in label_values]
        custom_lines = [plt.Line2D([], [], ls="", marker='.',
                                   mec='k', mfc=c, mew=.1, ms=20) for c in
                        colors]

        d2 = dict(labelMapNames)
        if Sn_inst != None:
            d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
                  loc='center left', bbox_to_anchor=(1.0, .5))

        if reductionMethod == "t-sne":
            fig.tight_layout()

        plt.show()

    def showTrainingPoints(self, Sn_inst=None, reductionMethod="pca"):

        X = self.training_dataset.X
        y = self.training_dataset.Y
        self.visualizePoints(self.training_dataset, Sn_inst, reductionMethod)

    def showNNLocality(self, Sn_inst, reductionMethod="pca", training=False):
        count_inst = self.explain_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.training_dataset_len), self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = gen_neighbors_info(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset[count_inst], self.starting_K,
                self.unique_filename, self.classifier, save=False)

        X = Kneighbors_data.X
        y = Kneighbors_data.Y
        self.visualizePoints(Kneighbors_data, Sn_inst, reductionMethod)

    def showNearestNeigh_type_2(self, Sn_inst, fig2, position,
                                reductionMethod="pca", training=False):

        from sklearn import decomposition

        count_inst = self.explain_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        # Plottarla con un colore diverso
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.training_dataset_len),
                self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = gen_neighbors_info(
                self.training_dataset,
                self.NearestNeighborsAll,
                self.explain_dataset[
                    count_inst], self.starting_K,
                self.unique_filename,
                self.classifier,
                save=False)

        X = Kneighbors_data.X
        y = Kneighbors_data.Y
        labelledInstance = deepcopy(instTmp2)

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(Kneighbors_data)):
                dataK.append(Kneighbors_data[k].list)

            columnsA = [i.name for i in Kneighbors_data.domain.variables]

            if Kneighbors_data.domain.metas != ():
                for i in range(0, len(Kneighbors_data.domain.metas)):
                    columnsA.append(Kneighbors_data.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in Kneighbors_data.domain.attributes]
            Xa = data[columnsA]
            y = Kneighbors_data.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True,
                             engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            istance_transformed = mca.transform(
                [[labelledInstance[i].value for i in
                  labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            XX = np.vstack([X, labelledInstance.x])
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            yy = np.concatenate((y, np.array([label_istance])))
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)



        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)
        label_istance = float(max(list(self.map_names_class.keys())) + 1)
        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            XX = np.vstack([X, istance_transformed])
            yy = np.concatenate((y, np.array([label_istance])))
            ax = fig2.add_subplot(1, 2, position, projection='3d')

            # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            label_values.append(int(label_istance))
            ax.set_title(self.classifier_name.upper())

        else:
            ax = fig2.add_subplot(1, 2, position)
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))
            ax.set_title(self.classifier_name.upper())

        colors = [sc.cmap(sc.norm(i)) for i in label_values]

        d2 = dict(labelMapNames)
        d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        # ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
        #          loc='center left', bbox_to_anchor=(0.9, .5), fontsize = 'x-small')

        return fig2, newdict, colors

    def showNNLocality_comparison(self, Sn_inst, fig2, position,
                                  reductionMethod="pca", training=False):
        count_inst = self.explain_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.training_dataset_len),
                self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = gen_neighbors_info(
                self.training_dataset,
                self.NearestNeighborsAll,
                self.explain_dataset[
                    count_inst], self.starting_K,
                self.unique_filename,
                self.classifier,
                save=False)

        return self.visualizePoints_comparison(Sn_inst, Kneighbors_data, fig2,
                                               position, reductionMethod,
                                               training)

    def visualizePoints_comparison(self, Sn_inst, datapoints, fig2, position,
                                   reductionMethod="pca", training=False):
        from sklearn import decomposition
        count_inst = self.explain_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)

        labelledInstance = deepcopy(instTmp2)
        X = datapoints.X
        y = datapoints.Y

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(datapoints)):
                dataK.append(datapoints[k].list)

            columnsA = [i.name for i in datapoints.domain.variables]

            if datapoints.domain.metas != ():
                for i in range(0, len(datapoints.domain.metas)):
                    columnsA.append(datapoints.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in datapoints.domain.attributes]
            Xa = data[columnsA]
            y = datapoints.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True,
                             engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            istance_transformed = mca.transform(
                [[labelledInstance[i].value for i in
                  labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            XX = np.vstack([X, labelledInstance.x])
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            yy = np.concatenate((y, np.array([label_istance])))
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)



        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)
        label_istance = float(max(list(self.map_names_class.keys())) + 1)
        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            XX = np.vstack([X, istance_transformed])
            yy = np.concatenate((y, np.array([label_istance])))
            ax = fig2.add_subplot(1, 2, position, projection='3d')

            # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            label_values.append(int(label_istance))
        else:
            ax = fig2.add_subplot(1, 2, position)
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))

        colors = [sc.cmap(sc.norm(i)) for i in label_values]
        custom_lines = [plt.Line2D([], [], ls="", marker='.',
                                   mec='k', mfc=c, mew=.1, ms=20) for c in
                        colors]

        d2 = dict(labelMapNames)
        d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
                  loc='center left', bbox_to_anchor=(0.9, .5),
                  fontsize='x-small')

        return fig2

    def showExplainDatasetTabularForm(self):
        return convertOTable2Pandas(self.explain_dataset,
                                    list(map(int, self.explain_indices)))

    def showMispredictedTabularForm(self, mispred_class=False):
        sel = self.getMispredicted(mispred_class=mispred_class)
        sel_index = [self.explain_indices.index(i) for i in sel]
        return convertOTable2Pandas(self.explain_dataset, list(map(int, sel)),
                                    sel_index, self.classifier,
                                    self.map_names_class)

    def showNearestNeighTabularForm(self, Sn_inst, training=False):
        count_inst = self.explain_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.training_dataset_len),
                self.K)
        if training == True:
            Kneighbors_data, labelledInstance = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, labelledInstance = gen_neighbors_info(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset[count_inst], self.starting_K,
                self.unique_filename, self.classifier, save=False)
        Kneigh_pd = convertOTable2Pandas(Kneighbors_data)
        return Kneigh_pd

    def interactiveMispredicted(self, mispred_class=False):
        from ipywidgets import HBox
        style = {'description_width': 'initial'}
        classes = ["All classes"] + self.classes[:]
        w = widgets.Dropdown(
            options=classes,
            description='Mispredicted classes',
            value="All classes",
            disabled=False, style=style
        )
        display(w)

        def clearAndShow(btNewObj):
            clear_output()
            display(w)
            display(h)

        def getMispredictedInteractiveButton(btn_object):
            if w.value == "All classes":
                sel = self.getMispredicted()
            else:
                sel = self.getMispredicted(mispred_class=w.value)
            sel_index = [self.explain_indices.index(i) for i in sel]
            misp = convertOTable2Pandas(self.explain_dataset,
                                        list(map(int, sel)),
                                        sel_index, self.classifier,
                                        self.map_names_class)
            from IPython.display import display
            display(misp.head())

        btnTargetC = widgets.Button(description='Get mispredicted')
        btnTargetC.on_click(getMispredictedInteractiveButton)
        btnNewSel = widgets.Button(description='Clear')
        btnNewSel.on_click(clearAndShow)
        h = HBox([btnTargetC, btnNewSel])
        display(h)



    #NEW_UPDATE
    # ************************************************************************************************ #
    def update_explain_instance(self, instance_explanation, rule_body_indices):
        target_class=instance_explanation.target_class
        instance=instance_explanation.instance
        c = self.classifier(instance, False)
        target_class_index = instance_explanation.instance_class_index
        pred = self.classifier(instance, True)[0][target_class_index]

        difference_map = instance_explanation.map_difference

        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        instance_predictions_cache = {}
        single_attribute_differences=instance_explanation.diff_single

        #Rule 1 element or already existing: no update needed
        if len(rule_body_indices) <= 1 or ','.join(map(str, rule_body_indices)) in difference_map:
            return instance_explanation

        PI_rel2, difference_map, error, impo_rules_complete = self.compute_prediction_difference_user_rule(
            rule_body_indices, instance,
            instance_predictions_cache,
            target_class, target_class_index, pred,
            single_attribute_differences, difference_map)



        instance_explanation = XPLAIN_explanation(self,
                                                  target_class,
                                                  instance,
                                                  single_attribute_differences,
                                                  instance_explanation.k,
                                                  error,
                                                  difference_map)

        return instance_explanation
    # ************************************************************************************************ #
    def compute_prediction_difference_user_rule(self, rule_body_indices, instance,
                          instance_predictions_cache, target_class,
                          target_class_index, pred, single_attribute_differences, difference_map):
        # Consider only rules with more than 1 attribute since we compute the differences
        # for single attribute changes already in compute_prediction_difference_single
        difference_map_key = ",".join(map(str, rule_body_indices))
        difference_map[difference_map_key] = compute_prediction_difference_subset(
                self.training_dataset, instance, rule_body_indices,
                self.classifier, target_class_index, instance_predictions_cache)

        impo_rules_complete=[list(map(int,e.split(","))) for e in list(difference_map.keys())]
        error_single, error, PI_rel2 = compute_error_approximation(self.mappa_class,
                                                                   pred,
                                                                   single_attribute_differences,
                                                                   impo_rules_complete,
                                                                   target_class,
                                                                   difference_map)


        return PI_rel2, difference_map, error, [impo_rules_complete]
    # ************************************************************************************************ #
    def getGlobalExplanationRules(self):
        import copy    
        global_expl=Global_Explanation(self)
        global_expl=global_expl.getGlobalExplanation()
        return global_expl

    

    # ************************************************************************************************ #

    




def get_KNN_threshold_max(KneighborsUser, len_dataset, thresholdError,
                          maxKNNUser):
    if KneighborsUser:
        k = int(KneighborsUser)
    else:
        import math
        k = int(round(math.sqrt(len_dataset)))

    if thresholdError:
        threshold = float(thresholdError)
    else:
        threshold = 0.10

    if maxKNNUser:
        max_n = int(maxKNNUser)
    else:
        max_n = getStartKValueSimplified(len_dataset)

    return k, threshold, max_n
