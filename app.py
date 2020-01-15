from copy import deepcopy

from flask import Flask, jsonify, abort, request
from flask_cors import CORS


from XPLAIN_class import XPLAIN_explainer
from XPLAIN_explanation_class import XPLAIN_explanation

from XPLAIN_user_explanation_class import User_Explanation

from XPLAIN_utils.XPLAIN_utils import *
app = Flask(__name__)
CORS(app)


classifiers = {
    'Naive Bayes': {'name': 'nb'},
    'Random Forest': {'name': 'rf'},
    'Neural Network': {'name': 'nn'},
    'K-Nearest Neighbor': {'name': 'knn'}
}


datasets = {
    'Zoo': {'file': 'zoo'},
    'Adult': {'file': 'datasets/adult_d.arff'},
    'Monks': {'file': 'monks-1'},
    'Monks-extended': {'file': 'datasets/monks_extended.arff'}
}

analyses = {'1explain': {"display_name": "Explain the prediction"},
            '4whatif': {"display_name": "What If analysis"},
            '2mispredicted': {"display_name": "Mispredicted analysis"},
            '3user_rules': {"display_name": "User defined rules"},
            "3explaination_comparison":{"display_name": "Explanation comparison"},
            "global_explanation":{"display_name": "Explanation metadata"},
            '2t_class_comparison':{"display_name": "Target class explanation comparison"}
            }

analyses_on_instance = {'4whatif': {"display_name": "What If analysis"},
            '3user_rules': {"display_name": "User defined rules"},
            "3explaination_comparison":{"display_name": "Explanation comparison"},
            '2t_class_comparison':{"display_name": "Target class explanation comparison"}
            }

global_analyses = {'1explain': {"display_name": "Explain the prediction of a new instance"},
            '2mispredicted': {"display_name": "Mispredicted analysis"},
            "global_explanation":{"display_name": "Explanation metadata"}
            }
# The application's global state.
# Initialized with default values to speed up development
state = {
    'dataset': 'zoo',
    'classifier': 'nb',
    'class': None,
    'instance': None,
    'explainer': XPLAIN_explainer('zoo', 'nb', random_explain_dataset=True),
    'user_explanation':None,
    'classifier2':None,
    'explainer2': None,
    'instance_id': None,
    'last_explanation':None,
    'proceed':False,
    'class1': None,
    'class2': None,
    'analysis_type': '1explain'
}

state["instance_id"]=state["explainer"].explain_indices[0]

# ************************************************************************************************ #

@app.route('/datasets')
def get_datasets():
    """GET /datasets returns all of the datasets"""
    return jsonify(list(datasets.keys()))


@app.route('/dataset/<name>', methods=['POST'])
def post_dataset(name):
    """POST /dataset/Zoo updates the local state setting the dataset"""
    #if name not in datasets:
    #    abort(404)
    if name in datasets:
        state['dataset'] = datasets[name]['file']
    elif (len(name.split("."))>1 and name.split(".")[1] in ["csv", "arff", "tab"]):
        state["dataset"]="./datasets/"+name
    else:
        state["dataset"]=name
    state["proceed"] = False
    return ""


# ************************************************************************************************ #

@app.route('/classifiers')
def get_classifers():
    """GET /classifiers returns all of the classifiers"""
    return jsonify(list(classifiers.keys()))


@app.route('/classifier/<name>', methods=['POST'])
def post_classifer(name):
    """POST /classifiers/Naive%20Bayes updates the local state setting the classifer"""
    if name not in classifiers:
        abort(404)
    state['classifier'] = classifiers[name]['name']
    state['explainer'] = None
    return ""

# ************************************************************************************************ #

@app.route('/classifiers_2')
def get_classifers_2():
    """GET /classifiers returns all of the classifiers"""
    return jsonify([k for k in classifiers if classifiers[k]["name"]!=state['classifier']])


@app.route('/classifier_2/<name>', methods=['POST'])
def post_classifer_2(name):
    """POST /classifiers/Naive%20Bayes updates the local state setting the classifer"""
    if name not in classifiers:
        abort(404)
    state['classifier2'] = classifiers[name]['name']
    state['explainer2'] = xp = getOrCompute_explainer(state['dataset'], state['classifier2'])
    return ""

# ************************************************************************************************ #
@app.route('/instances_class_comparison')
@app.route('/instances')
def get_instances():
    """
    GET /instances returns all of the instances (with their class) selected for explanation from the
    dataset which has been previously set with a POST /dataset/<name>. In this process it creates a
    new XPLAIN_explainer object, therefore reading/loading in memory the dataset.
    """
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    xp = state['explainer']
    d = xp.explain_dataset

    return jsonify({
        'domain': [(attr.name, attr.values) for attr in d.domain],
        'instances': [(list(instance.x) + list(instance.y), ix) for instance, ix in
                      zip(d, xp.explain_indices)],
        'classes': [*d.domain.class_var.values], 
        'analysis_type':state["analysis_type"]
    })


@app.route('/instance/<id>', methods=['POST'])
def post_instance(id):
    """POST /instance/17 updates the local state setting the instance"""
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    body = request.get_json(force=True)

    xp = state['explainer']
    state['instance'] = xp.explain_dataset[xp.explain_indices.index(id)]
    state['class'] = body['class']
    state['instance_id'] = id
    state['proceed'] = False
    return ""

# ************************************************************************************************ #
@app.route('/instances_class_comparison/<id>', methods=['POST'])
def post_instance_2(id):
    """POST /instance/17 updates the local state setting the instance"""
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    body = request.get_json(force=True)

    xp = state['explainer']
    state['instance'] = xp.explain_dataset[xp.explain_indices.index(id)]
    if "class" in body:
        state["class1"] = body['class']
    state['instance_id'] = id
    state['proceed'] = False
    if "class2" in body:
        state["class2"]=body["class2"]
    return ""
# ************************************************************************************************ #

@app.route('/class_comparison')
def get_target_classes():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    d = state['explainer'].explain_dataset
    return jsonify({
        'classes': [*d.domain.class_var.values],
        'class_1':state['class'] if state['proceed'] else False
    })

@app.route('/class_comparison/<name>', methods=['POST'])
def post_class_comparison(name):
    body = request.get_json(force=True)
    state['class1'] = body['class']
    state['class2'] = body['class2']
    return ""

# ************************************************************************************************ #

@app.route('/analyses_new')
def get_analyses_new():
    """GET /analyses returns all of the analyses"""
    return jsonify({"analyses_on_instance":analyses_on_instance, "global_analyses":global_analyses, "instance_id": state["instance_id"], "classifier_name":state["classifier"], "dataset_name": state["dataset"]})

@app.route('/analyses_new/<name>', methods=['POST'])
def post_analysis_new(name):
    state['proceed'] = True if name=="true" else False
    state['analysis_type']='1explain'
    return ""

# ************************************************************************************************ #

@app.route('/analyses')
def get_analyses():
    """GET /analyses returns all of the analyses"""
    return jsonify(analyses)

@app.route('/analyses/<name>', methods=['POST'])
def post_analyses(name):
    """POST /dataset/Zoo updates the local state setting the dataset"""
    if name not in analyses:
        abort(404)
    state['analysis_type'] = name
    return ""

# ************************************************************************************************ #

def explanation_to_dict(xp: XPLAIN_explanation):
    e: XPLAIN_explainer = xp.XPLAIN_explainer_o
    return {
        'instance': {attr.name: {'value': attr.values[int(value_ix)], 'options': attr.values} for
                     (attr, value_ix) in
                     zip(xp.instance.domain.attributes, xp.instance.x)},
        'domain': [(attr.name, attr.values) for attr in e.training_dataset.domain.attributes],
        'diff_single': xp.diff_single,
        'map_difference': xp.map_difference,# {k: v for k, v in xp.map_difference.items() if len(k)>1} #
        'k': xp.k,
        'error': xp.error,
        'target_class': xp.target_class,
        'instance_class_index': xp.instance_class_index,
        'prob': xp.prob, 
        'explainer_info': {"dataset_name": e.dataset_name, 
                            "classifier_name": [k for k,v in classifiers.items() if v["name"] == e.classifier_name][0],
                            "meta":"x="+xp.instance.metas[0] if xp.instance.metas else "x"},
        'true_class': xp.instance.get_class().value
        #'single_rules':{k: v for k, v in xp.map_difference.items() if len(k)==1}
    }


@app.route('/explanation')
def get_explanation():
    """
    GET /explanation returns the explanation for the instance w.r.t its class, using the explainer.
    The instance was previously set with POST /instance/<id> and belongs to the dataset set with
    POST /dataset/<name>. The explainer was created in a preceding call to POST /instance/<id>.
    """
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)

    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']


    class_ = instance.get_class().value if state['class'] is None else state['class']

    
    e=getOrCompute_explanation(instance, xp, class_)
        
    state["last_explanation"]=deepcopy(e)
    return jsonify(explanation_to_dict(e))

def getOrCompute_explanation(instance, xp, classname):
    e_name='expl_{0}_{1}_{2}_{3}'.format(xp.dataset_name, xp.classifier_name, state["instance_id"], classname)
    e=openPickle("./explanations", e_name)
    if e==False:
        e = xp.explain_instance(instance, target_class=classname)
        savePickle(e,"./explanations", e_name)
    return e

# ************************************************************************************************ #

@app.route('/whatIfExplanation', methods=['GET', 'POST'])
def get_what_if_explanation():
    """
    GET /whatIfExplanation returns the what-if explanation for the instance w.r.t its class, using the
    explainer.
    The instance was previously set with POST /instance/<id> and belongs to the dataset set with
    POST /dataset/<name>. The explainer was created in a preceding call to POST /instance/<id>.

    POST /whatIfExplanation allows to perturb the instance's attributes and get an explanation for
    the perturbed instance.
    """
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']

    
    if request.method!='POST' and state["proceed"]:
        e=state["last_explanation"]
    else:
        if request.method == 'POST':
            perturbed_attributes = request.get_json(force=True)

            perturbed_instance = deepcopy(instance)
            for k, v in perturbed_attributes.items():
                perturbed_instance[k] = v['options'].index(v['value'])

            instance = perturbed_instance
        class_ = instance.get_class().value if state['class'] is None else state['class']
        e = xp.explain_instance(instance, target_class=class_)

    a=jsonify(
        {'explanation': explanation_to_dict(e),
         'attributes': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i)
                        in
                        zip(instance.domain.attributes, instance.x)} })
    import time
    return jsonify(
        {'explanation': explanation_to_dict(e),
         'attributes': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i)
                        in
                        zip(instance.domain.attributes, instance.x)},
         'cnt_revision':int(time.time()) })

# ************************************************************************************************ #
@app.route('/user_rules', methods=['GET', 'POST'])
def get_user_rules_explanation():
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']
    attributes_names=[a.name for a in instance.domain.attributes]
    class_ = instance.get_class().value if state['class'] is None else state['class']

    if request.method == 'POST':
        rule_body_indices=[attributes_names.index(s_a)+1 for s_a in request.get_json(force=True)]
        if state['user_explanation'] is None:
            state['user_explanation']=User_Explanation(state["last_explanation"]) if state["proceed"] else User_Explanation(xp.explain_instance(instance, target_class=class_)) 
        e=xp.update_explain_instance(state['user_explanation'].instance_explanation, rule_body_indices)
        state['user_explanation'].updateUserRules(e,rule_body_indices)
    else:
        if state["proceed"]:
            e=state["last_explanation"]
        else:
            e=xp.explain_instance(instance, target_class=class_)
            state["last_explanation"]= deepcopy(e)
        state['user_explanation']= User_Explanation(e)
    return jsonify(
        {'explanation': explanation_to_dict(state['user_explanation'].instance_explanation),
         'attributes': attributes_names,#[a.name+"="+instance[a].value for (a,i) in zip(instance.domain.attributes, instance.x)]
         'id_user_rules': state['user_explanation'].id_user_rules
        })

# ************************************************************************************************ #

@app.route('/mispred_instances')
def get_MispredicedInstances():

    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    if state['explainer'] is None:
        state['explainer'] = xp = getOrCompute_explainer(state['dataset'], state['classifier']) 
        state['userI']=True

    xp=state['explainer']

    mispredicted_instances_df=xp.showMispredictedTabularForm()

    d=xp.explain_dataset
    attr_domain=[(attr.name, attr.values) for attr in d.domain]
    attr_domain.append(("pred", d.domain.class_var.values))

    mispred_instances=[]
    for index, row in mispredicted_instances_df.iterrows():
        map_i=row.to_dict()
        map_i["id"]=index
        mispred_instances.append(map_i)

    return jsonify({
        'domain': attr_domain,
        'mispred_instances': mispred_instances,
        'classes': [*d.domain.class_var.values]
    })


@app.route('/mispred_instance/<id>', methods=['POST'])
def post_MispredictedInstance(id):

    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    body = request.get_json(force=True)

    xp = state['explainer']
    state['instance'] = xp.explain_dataset[xp.explain_indices.index(id)]
    state['class'] = body['class']
    state['instance_id'] = id
    return ""

# ************************************************************************************************ #

@app.route('/explanation_comparison')
def get_explanation_comparison():
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']

    class_ = instance.get_class().value if state['class'] is None else state['class']

    e1 = state["last_explanation"] if state["proceed"] else getOrCompute_explanation(instance, xp, class_) 
    state["last_explanation"]=deepcopy(e1)
    xp2 = state['explainer2']

    e2=getOrCompute_explanation(instance, xp2, class_)


    return jsonify({"exp1":explanation_to_dict(e1), "exp2":explanation_to_dict(e2)})
# ************************************************************************************************ #

@app.route('/explanation_class_comparison')
def get_explanation_class_comparison():

    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']

    class_ = instance.get_class().value if state['class1'] is None else state['class1']

    e1 = state["last_explanation"] if (state["proceed"] and class_==state["class"]) else getOrCompute_explanation(instance, xp, class_)
    if (state["proceed"] and class_==state["class"]):
        state["last_explanation"]=deepcopy(e1)
    xp2 = state['explainer2']

    class2 = instance.get_class().value if state['class2'] is None else state['class2']

    e2=getOrCompute_explanation(instance, xp, class2)

    return jsonify({"exp1":explanation_to_dict(e1), "exp2":explanation_to_dict(e2)})


# ************************************************************************************************ #

@app.route('/global_explanation')
def get_global_explanation():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)
    xp = state['explainer']

    if state['explainer'] is None :
        state['explainer'] = xp = getOrCompute_explainer(state['dataset'], state['classifier'])

    global_e=openPickle("./global_explanations", 'global_expl_{0}_{1}'.format(xp.dataset_name, xp.classifier_name))
    if global_e==False:
        global_e= xp.getGlobalExplanationRules()
        savePickle(global_e, "./global_explanations", 'global_expl_{0}_{1}'.format(xp.dataset_name, xp.classifier_name))
    global_e_to_send=computeToSend(global_e, xp)
    return jsonify(global_e_to_send)
    

def computeToSend(global_e, xp):
    global_e_to_send={}
    for key in global_e.map_global_info:
        g_attr_contr=global_e.getGlobalAttributeContributionAbs(target_class=key)
        g_attr_value_contr=global_e.getGlobalAttributeValueContributionAbs(k=16,target_class=key)
        g_rule_contr=global_e.getGlobalRuleContributionAbs(target_class=key)
        rule_mapping=global_e.getRuleMapping(g_rule_contr, k=16)
        r_inv={v: k for k, v in rule_mapping.items()}
        g_rule_contr_k={r_inv[k]: g_rule_contr[k] for k in g_rule_contr if k in r_inv}
        global_e_to_send[key]= {
         'attribute_explanation': {'x': [i.name for i in xp.explain_dataset.domain.attributes],
         'y':[g_attr_contr[i.name] for i in xp.explain_dataset.domain.attributes]},
         'attribute_value_explanation': {'x': list(g_attr_value_contr.keys()),
         'y':list(g_attr_value_contr.values())},
          'rules_explanation': {'x': list(g_rule_contr_k.keys()),
         'y':list(g_rule_contr_k.values())},
          'rule_mapping': [k+" = { "+rule_mapping[k]+"}" for k in rule_mapping]}
    global_e_to_send['explainer_info']={"dataset_name": xp.dataset_name, 
                            "classifier_name": [k for k,v in classifiers.items() if v["name"] == xp.classifier_name][0], 
                            "target_classes": [*xp.explain_dataset.domain.class_var.values]
                            }
    return global_e_to_send

# ************************************************************************************************ #

@app.route('/show_instances')
def show_instances():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)
    if state['explainer'] is None:
        state['explainer'] = getOrCompute_explainer(state['dataset'], state['classifier'])
    xp = state['explainer']
    d = xp.explain_dataset

    return jsonify({
        'domain': [(attr.name, attr.values) for attr in d.domain],
        'instances': [(list(instance.x) + list(instance.y), ix) for instance, ix in
                      zip(d, xp.explain_indices)],
        "dataset_name": xp.dataset_name

    })

# ************************************************************************************************ #

def getOrCompute_explainer(dataset_name, classifier_name):
    xp_name='expl_{0}_{1}'.format(dataset_name.split("/")[-1].split(".")[0], classifier_name)
    xp=openPickle("./explainers", xp_name)
    if xp==False:
        xp = XPLAIN_explainer(dataset_name, classifier_name, random_explain_dataset=True)
        savePickle(xp,"./explainers", xp_name)
    return xp
