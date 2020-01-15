#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

import numpy as np
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter('ignore')
from XPLAIN_class import *
from Explanation_w import *
import seaborn as sns

sns.set_palette('muted')
sns.set_context("notebook", #font_scale=1.5,
				rc={"lines.linewidth": 2.5})


class XPLAIN_util:
	def __init__(self):
		self.XPLAIN_obj=None
		self.XPLAIN_obj_comparison=None

	def setXPLAIN_obj(self, x_obj):
		self.XPLAIN_obj=x_obj


	def getXPLAIN_obj(self):
		return self.XPLAIN_obj

	def setXPLAIN_obj_comparison(self, x_obj1, x_obj2):
		self.XPLAIN_obj_comparison=[x_obj1, x_obj2]


	def getXPLAIN_obj_comparison(self):
		return self.XPLAIN_obj_comparison

	def getXPLAIN_obj_comparison_1(self):
		if self.XPLAIN_obj_comparison!=None:
			return self.XPLAIN_obj_comparison[0] 
		return None

	def getXPLAIN_obj_comparison_2(self):
		if self.XPLAIN_obj_comparison!=None:
			return self.XPLAIN_obj_comparison[1] 
		return None


	def interactiveDatasetClassifierSelection(self):
		from ipywidgets import Button, HBox, VBox
		dataset_names=["monks-1","zoo", "adult"]
		classifier_names=["Random Forest","MLP-NN", "Naive Bayes"]
		dataset_names_map={"monks-1":"monks-1","zoo":"zoo", "adult":"datasets/adult_d.arff", "COMPAS":"datasets/compas-scores-two-years_d.arff"}
		classifier_names_map={"Random Forest":"rf","MLP-NN":"nn", "Naive Bayes":"nb"}
		w1=widgets.Dropdown(
		options=dataset_names,
		description='Dataset',
		value="monks-1",
		disabled=False
		)
		w2=widgets.Dropdown(
		options=classifier_names,
		description='Classifier',
		value="Random Forest",
		disabled=False
		)
		hClasses=VBox([w1, w2])
		l=widgets.Label(value='Select dataset and classifier:')
		display(l)
		display(hClasses)
		def clearAndShow(btNewObj):
			clear_output()
			display(l)
			display(hClasses)
			display(h)
		def getXPLAIN(btn_object):
			e=XPLAIN_explainer(dataset_names_map[w1.value], [], classifier_names_map[w2.value], random_explain_dataset=True)
			self.setXPLAIN_obj(e)

		btnTargetC = widgets.Button(description='Select')
		btnTargetC.on_click(getXPLAIN)
		display(btnTargetC)


	def interactiveComparisonDatasetClassifierSelection(self):
		from ipywidgets import Button, HBox, VBox
		dataset_names=["monks-1","zoo", "adult"]
		classifier_names=["Random Forest","MLP-NN", "Naive Bayes"]
		dataset_names_map={"monks-1":"monks-1","zoo":"zoo", "adult":"datasets/adult_d.arff", "COMPAS":"datasets/compas-scores-two-years_d.arff"}
		classifier_names_map={"Random Forest":"rf","MLP-NN":"nn", "Naive Bayes":"nb"}
		w1=widgets.Dropdown(
		options=dataset_names,
		description='Dataset',
		value="monks-1",
		disabled=False
		)
		w2=widgets.Dropdown(
		options=classifier_names,
		description='Classifier 1',
		value="Random Forest",
		disabled=False
		)
		w3=widgets.Dropdown(
		options=classifier_names,
		description='Classifier 2',
		value="Naive Bayes",
		disabled=False
		)
		hClasses=VBox([w1, w2, w3])
		l=widgets.Label(value='Select dataset and classifier:')
		display(l)
		display(hClasses)
		def clearAndShow(btNewObj):
			clear_output()
			display(l)
			display(hClasses)
			display(h)
		def getXPLAIN(btn_object):
			e1=XPLAIN_explainer(dataset_names_map[w1.value], [], classifier_names_map[w2.value], random_explain_dataset=True)
			e2=XPLAIN_explainer(dataset_names_map[w1.value], [], classifier_names_map[w3.value], random_explain_dataset=True)
			self.setXPLAIN_obj_comparison(e1, e2)

		btnTargetC = widgets.Button(description='Select')
		btnTargetC.on_click(getXPLAIN)
		display(btnTargetC)


	def def_XPLAIN_o(self, d_name, classif_name):
		e=XPLAIN_explainer(d_name, [], classif_name, random_explain_dataset=True)
		self.setXPLAIN_obj(e)
		return e

	def def_XPLAIN_o_comparison(self, d_name, classif_name, classif_name2):
		e1=XPLAIN_explainer(d_name, [], classif_name, random_explain_dataset=True)
		e2=XPLAIN_explainer(d_name, [], classif_name2, random_explain_dataset=True)
		self.setXPLAIN_obj_comparison(e1, e2)
		return [e1, e2]
