from XPLAIN_class import XPLAIN_explainer


def get_explanation(dataset, classifier):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        explainer = XPLAIN_explainer(dataset, classifier, random_explain_dataset=True)
        instance = explainer.explain_dataset[0]
        return explainer.explain_instance(instance, target_class=instance.get_class().value)


if __name__ == "__main__":
    get_explanation("datasets/adult_d.arff", "rf")
