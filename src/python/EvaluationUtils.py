import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


class EvaluationUtils:

    @staticmethod
    def evaluateBinaryClassifier(actual_labels, predicted_probabilites):
        if(actual_labels is None or predicted_probabilites is None):
            raise ValueError(
                "actual_labels & predicted_probabilites need to be passed")

        num_samples = np.size(actual_labels)
        predcited_classes = [1 if proba >
                             0.5 else 0 for proba in predicted_probabilites]
        classification_metrics = classification_report(
            actual_labels, predcited_classes)
        auc = roc_auc_score(actual_labels, predicted_probabilites)

        print("# Samples : " + str(num_samples))
        print("# Classification Metrics : ")
        print("AUC : " + str(auc))
        print(classification_metrics)

        return auc

    @staticmethod
    def classifier_AUC_cv_eval(model):
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
