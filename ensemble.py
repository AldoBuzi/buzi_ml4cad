import copy
import numpy as np
from joblib import dump, load
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer, accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score, brier_score_loss


from itertools import chain, repeat, count, islice
from collections import Counter

def build_ensemble(models, path):
    ensemble = []
    for m in models:
        ensemble.append((m, load(path+f"{m}.joblib")))
    
    return ensemble


def predict_ensemble(ensemble, X, y, threshold=0.5):
    y_proba = []
    for m in ensemble:
        y_proba.append(m.predict_proba(X))
    y_proba = np.mean(y_proba, axis=0)
    y_pred = y_proba[:, 1] > threshold
    
    return y_proba, y_pred


def evaluate_ensemble(ensemble, X, y, threshold=0.5, verbose=True):
    y_proba, y_pred = predict_ensemble(ensemble, X, y)
    if verbose:
        print(classification_report(y, y_pred, digits=3))
        print(f"auroc {roc_auc_score(y, y_proba[:, 1]):.3f}")
        print(f"brier {brier_score_loss(y, y_proba[:, 1]):.3f}")
        print(confusion_matrix(y, y_pred))
  
    return (roc_auc_score(y, y_proba[:, 1]),f1_score(y, y_pred))


def find_best_ensemble(ensemble, X_valid, y_valid):
    results = []
    sum = 0
    while len(ensemble) > 1:
        tmp_res = []
        for m in ensemble:
            tmp = copy.copy(ensemble)
            tmp.remove(m)
            sum += 1
            names = [_name for _name, _m in tmp]
            tmp = [_m for _name, _m in tmp]
            acc = evaluate_ensemble(tmp, X_valid, y_valid, verbose=False)
            results.append((names, tmp, acc))
            tmp_res.append((m, acc))

        m, _ = max(tmp_res, key=lambda item:item[1])
        ensemble.remove(m)
        print(f"Somma totale {sum}")
        print(m)
        
    return max(results, key=lambda item:item[2])

def find_best_ensemble2(models_list, path, X_valid, y_valid, verbose = False):
    results = []
    for key in range(1,len(models_list)):
        combinations = list(unique_combinations(models_list,key))
        for combine in combinations:
            ensemble = build_ensemble(combine,path)
            tmp = [_m for _, _m in ensemble]
            acc = evaluate_ensemble(tmp, X_valid, y_valid, verbose= verbose)
            results.append((combine, tmp, acc))
    results.sort(key=lambda item:-item[2][1])
    results = results[:5]
    copy = results.copy()
    filter = [(x[0],x[2]) for x in results]
    copy.sort(key=lambda item:-item[2][0])
    for names, model, score  in copy[:5]:
        if (names, score) not in filter:
            results.append((names,model,score))
    results.sort(key=lambda item:-item[2][1])
    index = 1
    for ensemble in results[:10]:
        names, model, score = ensemble
        print("##############################################")
        print (f" Rank: #{index} Names: {names}, Score: {score}")
        index +=1
    #return model ensemble
    return results
    

def repeat_chain(values, counts):
    return chain.from_iterable(map(repeat, values, counts))


def unique_combinations_from_value_counts(values, counts, r):
    n = len(counts)
    indices = list(islice(repeat_chain(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), repeat_chain(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), repeat_chain(count(j), counts[j:])):
            indices[i] = j


def unique_combinations(iterable, r):
    values, counts = zip(*Counter(iterable).items())
    return unique_combinations_from_value_counts(values, counts, r)