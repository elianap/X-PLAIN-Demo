#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from XPLAIN_utils.LACE_utils.LACE_utils1 import *
from XPLAIN_class import *
from XPLAIN_class import *


def getPerc(input_per):
    result_mean_instance={}
    for k in input_per.keys():
        if list(input_per[k].keys())[0]!="":
            result_mean_instance[k]=sum([input_per[k][k2] for k2 in input_per[k]])/len(([input_per[k][k2] for k2 in input_per[k]]))
    mean_per=sum(result_mean_instance[k] for k in result_mean_instance)/len(result_mean_instance)
    result_mean_instanceA={}
    for k in input_per.keys():
        result_mean_instanceA[k]=sum([input_per[k][k2] for k2 in input_per[k]])/len(([input_per[k][k2] for k2 in input_per[k]]))
    mean_per_A=sum(result_mean_instanceA[k] for k in result_mean_instanceA)/len(result_mean_instanceA)
    return mean_per, mean_per_A


def getmap_instance_NofKNNIterationInfo(map_instance_NofKNN):
    sum=0
    for k in map_instance_NofKNN.keys():
        sum=sum+map_instance_NofKNN[k]

    avg=float(sum)/float(len(map_instance_NofKNN))
    minv=min([map_instance_NofKNN[k] for k in map_instance_NofKNN])
    maxv=max([map_instance_NofKNN[k] for k in map_instance_NofKNN])
    return avg, minv, maxv 


def getInfoError(rel, rel1, d, t):
    avg,minv,maxv=getmap_instance_NofKNNIterationInfo(rel)
    avg1,minv1,maxv1=getmap_instance_NofKNNIterationInfo(rel1)
    diff=avg1-avg
    print("Approximation Gain:", diff)


def printImpoRuleInfoOLD(instT, NofKNN, out_data,map_difference,impo_rules_complete, impo_rules):
	print(instT,":  K=", NofKNN, "Rules", impo_rules)
	print("PredDifference", map_difference, out_data, "\n")


def compareLocalityLOLD(le1, le2, Sn_inst, reductionMethod="pca",training=False):
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    fig2, labeltest1, colors1=le1.showNearestNeigh_type_2( Sn_inst,fig2,1,reductionMethod)
    fig2, labeltest2, colors2=le2.showNearestNeigh_type_2( Sn_inst, fig2,2, reductionMethod)
    labeltest1.update(labeltest2)
    label_values=list(labeltest1.keys())
    from collections import OrderedDict
    colors=colors1+colors2
    cmap = plt.cm.Spectral
    norm=plt.Normalize()
    colors=list(OrderedDict.fromkeys(colors))

    custom_lines = [plt.Line2D([],[], ls="", marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]
    fig2.legend(custom_lines, [lt[1] for lt in labeltest1.items()], loc='center left', bbox_to_anchor=(0.9, .5), fontsize = 'x-small')
    plt.tight_layout()
    plt.show()

def compareLocalityL(le1, le2, Sn_inst, reductionMethod="mca",training=False):
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    fig2, labeltest1, colors1=le1.showNearestNeigh_type_2( Sn_inst,fig2,1,reductionMethod)
    fig2, labeltest2, colors2=le2.showNearestNeigh_type_2( Sn_inst, fig2,2, reductionMethod)
    labeltest1.update(labeltest2)
    label_values=list(labeltest1.keys())
    from collections import OrderedDict
    colors=colors1+colors2
    cmap = plt.cm.Spectral
    norm=plt.Normalize()
    colors=list(OrderedDict.fromkeys(colors))
    custom_lines = [plt.Line2D([],[], ls="", marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]    
    if reductionMethod=="pca" or  reductionMethod=="mca":
        reductionMethod=reductionMethod.upper()
        fig2.get_axes()[0].legend(custom_lines, [lt[1] for lt in labeltest1.items()],loc='lower left', bbox_to_anchor= (0,-0.05,1,0.35), ncol=10, mode="expand", borderaxespad=0)
    else:
        reductionMethod="t-SNE"
        fig2.get_axes()[0].legend(custom_lines, [lt[1] for lt in labeltest1.items()],loc='lower left', bbox_to_anchor= (0,-0.1,1,0.35), ncol=10, mode="expand", borderaxespad=0)

    fig2.suptitle("Locality Inspection - "+reductionMethod, size=12)
    plt.tight_layout()
    plt.show()


def compareLocality(le1, le2, Sn_inst, reductionMethod="pca",training=False):
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    fig2=le1.showNNLocality_comparison(Sn_inst,fig2,1,reductionMethod,training)
    fig2=le2.showNNLocality_comparison(Sn_inst,fig2,2,reductionMethod,training)
    plt.tight_layout()
    plt.show()

def compareLocalityTrueLabels(le1, Sn_inst, reductionMethod="pca",training=False):
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    fig2=le1.showNNLocality_comparison(Sn_inst,fig2,1,reductionMethod,training)
    fig2=le1.showNNLocality_comparison(Sn_inst,fig2,2,reductionMethod,training=True)
    plt.tight_layout()
    plt.show()


def convertOTable2Pandas(orangeTable, ids=None, sel="all", cl=None, mapName=None):
    import pandas as pd

    if sel=="all":
        dataK=[orangeTable[k].list for k in range(0,len(orangeTable))]
    else:
        dataK=[orangeTable[k].list for k in sel]

    columnsA=[i.name for i in orangeTable.domain.variables]

    if orangeTable.domain.metas!=():
        for i in range(0,len(orangeTable.domain.metas)):
            columnsA.append(orangeTable.domain.metas[i].name)
    data = pd.DataFrame(data=dataK, columns=columnsA)


    if cl!=None and sel!="all" and mapName!=None:
        y_pred=[mapName[cl(orangeTable[k], False)[0]] for k in sel]
        data["pred"]=y_pred

    if ids!=None:
        data["instance_id"]=ids
        data=data.set_index('instance_id')

    return data

def getExtractedRulesPrintFriendly(instT, impo_rules_complete):
    rulesPrint=[]
    unionRulePrint=[]
    for r in impo_rules_complete:
        rule="{"
        for k in r:
            rule=rule+instT[k-1].variable.name+"="+instT[k-1].value+", "
        rule=rule[:-2]+"}"
        if r==list(max(impo_rules_complete, key=len)) and len(impo_rules_complete)>1:
            unionRulePrint=rule
        else:
            rulesPrint.append(rule)
    return rulesPrint, unionRulePrint







def printImpoRuleInfo(instID, instT, NofKNN, out_data,map_difference,impo_rules_c, impo_rules):
    #print("ID: ", instID,"  K=", NofKNN)#, "Rules", impo_rules)
    #print("PredDifference", single_attribute_differences, difference_map, "\n")
    """
    rulesPrint, unionRulePrint=getExtractedRulesMapping(instT, impo_rules, list(difference_map.keys()), sep=", ")
    rulesPrint=list(rulesPrint.values())
    unionRulePrint=list(unionRulePrint.values())
    if rulesPrint!=[]:
        print("\nLocal rules:")
        for r in rulesPrint:
            print(r)
    if unionRulePrint!=[]:
        print("Union of rule bodies",unionRulePrint[0])
    """
    a=1



#input: imporules complete, impo_rules, separatore
def getExtractedRulesMapping(instT, impo_rules, impo_rules_c, sep=", "):
    rulesPrint={}
    unionRulePrint={}
    for r in impo_rules:
        #impo_rule
        if type(r)==str:
            rule="{"
            for k in r.split(sep):
                rule=rule+instT[int(k)-1].variable.name+"="+instT[int(k)-1].value+", "
            rule=rule[:-2]+"}"
            rulesPrint[r.replace(sep, ",")]=rule
    if impo_rules_c!=[]:
        union=max(impo_rules_c, key=len)
        if union.replace(",", ", ") not in impo_rules:
            rule="{"
            for k in union.split(","):
                rule=rule+instT[int(k)-1].variable.name+"="+instT[int(k)-1].value+", "
            rule=rule[:-2]+"}"
            unionRulePrint[union]=rule

    return rulesPrint, unionRulePrint


#input: imporules complete, impo_rules, separatore
def getExtractedRulesMapping_old(instT, impo_rules, impo_rules_c, sep=", "):
    rulesPrint={}
    unionRulePrint={}
    for r in impo_rules:
        #impo_rule
        if type(r)==str:
            rule="{"
            for k in r.split(sep):
                rule=rule+instT[int(k)-1].variable.name+"="+instT[int(k)-1].value+", "
            rule=rule[:-2]+"}"
            rulesPrint[r.replace(sep, ",")]=rule
    if len(impo_rules_c)>1:
        union=max(impo_rules_c, key=len)
        if union.replace(",", ", ") not in impo_rules:
            rule="{"
            for k in union.split(","):
                rule=rule+instT[int(k)-1].variable.name+"="+instT[int(k)-1].value+", "
            rule=rule[:-2]+"}"
            unionRulePrint[union]=rule


    return rulesPrint, unionRulePrint

def printMapping(instT, impo_rules, impo_rules_c, sep=", "):
    rulesPrint, unionRulePrint=getExtractedRulesMapping(instT, impo_rules, impo_rules_c,sep)
    if rulesPrint!={}:
        print("\nLocal rules:")
        for r in sorted(rulesPrint, key=len):
            print(r, " -> ", rulesPrint[r])
    if unionRulePrint!={}:
        print("Union of rule bodies")
        for r in unionRulePrint:
            print(r, " -> ", unionRulePrint[r])

def printMapping_v5(instT, impo_rules, impo_rules_c, y_label_mapping, sep=", "):
    rulesPrint, unionRulePrint=getExtractedRulesMapping(instT, impo_rules, impo_rules_c,sep)
    if rulesPrint!={}:
        print("\nLocal rules:")
        for r in sorted(rulesPrint, key=len):
            print(y_label_mapping[r], " -> ", rulesPrint[r])
    if unionRulePrint!={}:
        print("Union of rule bodies:")
        for r in unionRulePrint:
            print(y_label_mapping[r], " -> ", unionRulePrint[r])



def interactiveModelComparison(le1, le2, instID):
    from ipywidgets import Button, HBox, VBox
    from IPython.display import display
    import ipywidgets as widgets
    from IPython.display import clear_output

    classes1=["predicted", "trueLabel"]+le1.classes[:]
    wcl1=widgets.Dropdown(options=classes1, description='1º', value="predicted", disabled=False)
    classes2=["predicted", "trueLabel"]+le2.classes[:]
    wcl2=widgets.Dropdown(options=classes2, description='2º', value="predicted", disabled=False)
    hClasses=VBox([wcl1, wcl2])
    l=widgets.Label(value='Select models and target classes:')
    display(l)
    display(hClasses)
    def clearAndShow(btNewObj):
        clear_output()
        display(l)
        display(hClasses)
        display(h)

    def getExplainInteractiveButton(btn_object):
        getModelExplanationComparison(instID, le1, le2, wcl1.value, wcl2.value)
        
    btnTargetC = widgets.Button(description='Compute')
    btnTargetC.on_click(getExplainInteractiveButton)
    btnNewSel = widgets.Button(description='Clear')
    btnNewSel.on_click(clearAndShow)
    h=HBox([btnTargetC, btnNewSel])
    display(h)


def getModelExplanationComparison(Sn_inst, le1, le2, targetClass1, targetClass2):
    fig2 =  plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig2.add_subplot(1, 2, 1)    
    explanation_1, ax1=le1.getExplanation_i_axis( ax1, Sn_inst, targetClass1)
    ax2 = fig2.add_subplot(1, 2, 2)    
    explanation_2, ax2=le2.getExplanation_i_axis( ax2, Sn_inst,targetClass2)
    plt.tight_layout()
    plt.show()
    #return explanation_1, explanation_2




def interactiveModelComparisonInstance(le1, le2):
    from ipywidgets import Button, HBox, VBox
    from IPython.display import display
    import ipywidgets as widgets
    from IPython.display import clear_output

    classes1=["predicted", "trueLabel"]+le1.classes[:]
    wcl1=widgets.Dropdown(options=classes1, description='Class 1º', value="predicted", disabled=False)
    classes2=["predicted", "trueLabel"]+le2.classes[:]
    wcl2=widgets.Dropdown(options=classes2, description='Class 2º', value="predicted", disabled=False)
    
    w_ID=widgets.Dropdown(
    options=le1.n_insts,
    description="ID",
    disabled=False
    )
    hClasses=VBox([w_ID, wcl1, wcl2])
    l=widgets.Label(value='Select instance and target classes:')
    display(l)
    display(hClasses)
    def clearAndShow(btNewObj):
        clear_output()
        display(l)
        display(hClasses)
        display(h)

    def getExplainInteractiveButton(btn_object):
        getModelExplanationComparison(w_ID.value, le1, le2, wcl1.value, wcl2.value)
        
    btnTargetC = widgets.Button(description='Compute')
    btnTargetC.on_click(getExplainInteractiveButton)
    btnNewSel = widgets.Button(description='Clear')
    btnNewSel.on_click(clearAndShow)
    h=HBox([btnTargetC, btnNewSel])
    display(h)


def savePickle(model, dirO, name):
    import pickle
    createDir(dirO)
    with open(dirO+"/"+name+'.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    import os.path
    from os import path
    if path.exists(dirO+"/"+name+'.pickle'):
        with open(dirO+"/"+name+'.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return False