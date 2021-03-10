import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import app_feature_calc

#inheriting from base classes to get get_params, set_params,
class GroupImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.
    
    Parameters
    ----------    
    group_cols : list
    List of columns used for calculating the aggregated value 
    target : str
    The name of the column to impute
    metric : str
    The metric to be used for remplacement, can be one of ['mean', 'median']
    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''
    def __init__(self, group_cols, target, metric='mean'):
        
        assert metric in ['mean', 'median'], 'Unrecognized value for metric, should be mean/median'
        assert type(group_cols) == list, 'group_cols should be a list of columns'
        assert type(target) == str, 'target should be a string'
        
        self.group_cols = group_cols
        self.target = target
        self.metric = metric
    
    def fit(self, X, y=None):
        
        assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        
        impute_map = X.groupby(self.group_cols)[self.target].agg(self.metric).reset_index(drop=False)
        self.impute_map_ = impute_map
        
        return self 
    
    def transform(self, X, y=None):
        
        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map_')
        
        X = X.copy()
        
        for index, row in self.impute_map_.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind] = X.loc[ind].fillna(row[self.target])
        
        return X.values

    
def get_pipeline():
    # transformer for categorical features
    categorical_features_str = ['marital_status','faf_flag'] 
    categorical_transformer_str = Pipeline(
        [
            ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

    categorical_features_obj = ['first_generation'] 
    categorical_transformer_obj = Pipeline(
        [
            ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

    # transformer for numerical features
    numeric_features_md = ['new_age','distance','decision_days','high_school_gpa'] 
    numeric_transformer_md = Pipeline(
        [
            ('imputer_num', SimpleImputer(strategy = 'mean')),
            ('scaler', StandardScaler())
        ]
    )

    numeric_features_zr = ['athl_ofr','grnt_ofr','loan_ofr','schl_ofr','waiv_ofr','pell_ofr']
    numeric_transformer_zr = Pipeline(
        [
            ('imputer_num', SimpleImputer(strategy = 'constant', fill_value = 0)),
            ('scaler', StandardScaler())
        ]
    )

    numeric_feature_custom = []
    numeric_transformer_custom = Pipeline(
        [
            ('num_custom', GroupImputer(group_cols=['distance'], 
                                        target='efc', 
                                        metric='median')),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = 
        [
            ('categoricals_str', categorical_transformer_str, categorical_features_str), 
            ('categoricals_obj', categorical_transformer_obj, categorical_features_obj),
            ('numericals_md', numeric_transformer_md, numeric_features_md),
            ('numericals_zr', numeric_transformer_zr, numeric_features_zr),
            ('custom_impute',numeric_transformer_custom, numeric_feature_custom)

        ]
    )

    pipe = Pipeline(
        [
            ('preprocessing', preprocessor),
            ('clf', RandomForestClassifier())
        ]
    )

    return pipe


if __name__=="__main__":
    pipe = get_pipeline()
    df_adm = app_feature_calc.clean()
    x = df_adm.loc[:, df_adm.columns != 'enr_flag']
    y = df_adm.loc[:, df_adm.columns == 'enr_flag']
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


    param_grid = {
                    'clf__n_estimators': (100,150),
                    'clf__max_depth': [300, 500]
        }

    grid_clf= GridSearchCV(
            pipe,
            param_grid,
            cv=10
        )

    # # # print(pipe.get_params().keys()) # this command is used for checking param_grid parameters.
    grid_clf.fit(x_train,y_train.values.ravel())
    ytest_lst = y_test.values.tolist()
    y_pred_rf = grid_clf.predict(x_test)

    df_cf = pd.DataFrame(
        confusion_matrix(ytest_lst, y_pred_rf),
        columns=['Predicted Not Enroll', 'Predicted Enroll'],
        index=['True Not Enroll', 'True Enroll']
    )
    print(df_cf)
    print("Accuracy of random forest: %.3f" %metrics.accuracy_score(y_test, y_pred_rf))

