from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
import numpy as np
def get_preprocess_std_num(feat_names):
    """Preprocess only the numerical features."""

    def update_num_feats(x):
        if x in feat_names:
            return feat_names.index(x)
    # standardize these variables
    feat_names_num = ["Age", "fe", "Vessels","TSH","ft3","ft4", "Total cholesterol", "HDL","LDL","Triglycerides","Creatinina"]
    num_feat_index = list(map(update_num_feats, feat_names_num))
    num_feat_index = [x for x in num_feat_index if x is not None]
    preprocess_std_num = ColumnTransformer(
                                transformers = [('stand', StandardScaler(), num_feat_index)], 
                                remainder="passthrough",
                                verbose_feature_names_out=False
                            )
    return preprocess_std_num


def datasetSampler(
        model_name,
        model, 
        overSampler, 
        sampling_strategy,
        X_train, 
        y_train, 
        X_valid, 
        y_valid, 
        useUnderSampler =False):
    scores = []
    if useUnderSampler:
        under = RandomUnderSampler(sampling_strategy=sampling_strategy)
        X_train_sample, y_train_sample = under.fit_resample(X_train, y_train)
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    X_train_sample, y_train_sample = overSampler.fit_resample(X_train_sample, y_train_sample)
    for _ in range(5):
        model.fit(X_train_sample, y_train_sample)
        score = f1_score(y_valid, model.predict(X_valid), average="macro")
        scores.append(score)
    score = np.mean(scores)
    return (score,X_train_sample,y_train_sample, model)