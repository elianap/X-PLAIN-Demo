#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

import numpy as np
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter('ignore')
from XPLAIN_class import *
import seaborn as sns

sns.set_palette('muted')
sns.set_context("notebook", #font_scale=1.5,
                rc={"lines.linewidth": 2.5})


class Explanation_w:
    def __init__(self):
    	self.explanation=None

    def interactiveGetExplanation(self, XPLAIN_obj, mispredicted=False):
	    from ipywidgets import Button, HBox, VBox
	    classes=["predicted", "trueLabel"]+XPLAIN_obj.classes[:]
	    w_Target=widgets.Dropdown(
	    options=classes,
	    description='Target class',
	    value="predicted",
	    disabled=False
	    )
	    if mispredicted:
	    	ids_d=XPLAIN_obj.mispredictedInstances
	    	name='ID mispredicted'
	    else:
	    	ids_d=XPLAIN_obj.n_insts
	    	name='ID'
	    w_ID=widgets.Dropdown(
	    options=ids_d,
	    description=name,
	    disabled=False
	    )
	    hClasses=VBox([w_ID, w_Target])
	    l=widgets.Label(value='Select instance to be explained and target class:')
	    display(l)
	    display(hClasses)
	    def clearAndShow(btNewObj):
	        clear_output()
	        display(l)
	        display(hClasses)
	        display(h)

	    def getExplainInteractiveButton(btn_object):
	        expz=XPLAIN_obj.explain_instance(w_ID.value, target_class=w_Target.value)
	        self.setExpl(expz)
	    btnTargetC = widgets.Button(description='Compute')
	    btnTargetC.on_click(getExplainInteractiveButton)
	    btnNewSel = widgets.Button(description='Clear')
	    btnNewSel.on_click(clearAndShow)
	    h=HBox([btnTargetC, btnNewSel])
	    display(h)



    def setExpl(self, exp):
    	self.explanation=exp



    def getExpl(self):
    	return self.explanation



    def def_Explanation(self, XPLAIN_obj, targetClass="predicted", mispredicted=False, mispred_class=False):
    	if mispredicted:
    		id_misp=XPLAIN_obj.getMispredicted( mispred_class=mispred_class)[0]
    		expz=XPLAIN_obj.explain_instance(str(id_misp), target_class=targetClass)
    	else:
    		expz=XPLAIN_obj.explain_instance(str(XPLAIN_obj.n_insts[0]), target_class=targetClass)
    	self.setExpl(expz)
    	return expz