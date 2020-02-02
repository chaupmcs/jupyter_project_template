from sklearn.model_selection import train_test_split,  KFold
import lightgbm as lgb
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def lgb_regression(data, feature_cols, cate_cols,  regress_params = None,
                       early_stop = 500, num_splits = 5, label = "label", n_cores = 8, verbose = 500,
                       is_printing_features = False, hyper_params_tuning = False):
                     
    '''  
    copy these lines of code in your notebook:
    
            data = df
            feature_cols = feature_cols
            cate_cols = cate_cols
            early_stop = 500
            num_splits = 5
            verbose = 500
            n_cores = 4
            random_seed = 912
            label = "label"

            regress_params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':0.5,
                           'importance_type':'split', 'learning_rate':0.01, 'max_depth':5,
                           'min_child_samples':30, 'min_child_weight':0.001, 'min_split_gain':0.00,
                           'n_estimators':30000, 'n_jobs':n_cores, 'num_leaves':32, 'objective':"rmse", "metric": "rmse",
                           'random_state':random_seed, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
                           'subsample':0.5, 'subsample_for_bin':200000, 'subsample_freq':10}

            res = run_lgb_regression(data, feature_cols, cate_cols, regress_params = regress_params, 
                       early_stop = early_stop, num_splits = num_splits, label = label,
                       n_cores = n_cores, verbose = verbose,
                       is_printing_features = False, hyper_params_tuning = False)
    '''
    import warnings
    warnings.filterwarnings('ignore')
    start_time = datetime.now()
    
     ## prepare the model
    default_params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':0.5,
              'importance_type':'split', 'learning_rate':0.01, 'max_depth':5,
               'min_child_samples':30, 'min_child_weight':0.001, 'min_split_gain':0.00,
               'n_estimators':30000, 'n_jobs':n_cores, 'num_leaves':32, 'objective':"rmse", "metric": "rmse",
               'random_state':912, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
               'subsample':0.5, 'subsample_for_bin':200000, 'subsample_freq':10}
  
        
    if regress_params is not None:
        default_params.update(regress_params)

   
    random_seed =  default_params['random_state']
    cv = KFold(n_splits = num_splits, random_state = random_seed, shuffle=True)

   

    oof_list = []
    feature_importance_lgbm = []
    model_list = []

    if hyper_params_tuning == False:
        print("num features = {}".format(len(feature_cols) ))
        
    if is_printing_features:
          [print(c) for c in feature_cols]

    for i, (train_index, valid_index) in enumerate(cv.split(data)):
        regress_model = lgb.LGBMRegressor(**default_params)
        X = data[feature_cols]
        y = data[label]

        # Create data for this fold
        X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[valid_index,:].copy()
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[valid_index].copy()
        if hyper_params_tuning == False:
            print( "\n...running fold {}/{}".format(i+1, num_splits))
        
        
        record_store = dict()
        regress_model.fit(X_train, y_train, feature_name = feature_cols,  categorical_feature = cate_cols,
                        early_stopping_rounds = early_stop,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_names=["train", "valid"],
                        verbose = verbose, callbacks = [lgb.record_evaluation(record_store)])


        # feature importance of this fold:
        feature_importance_lgbm.append(regress_model.feature_importances_)
        y_pred = regress_model.predict(X_valid)
      
        tmp = pd.concat([X_valid, y_valid], 1)
        tmp['pred'] = y_pred
        tmp["uid"] = data[["src_id"]].iloc[valid_index,:].copy()
        oof_list.append(tmp)
        
        # plot learning curve 
        # f = lgb.plot_metric(record_store, figsize=(10,8))
      
        model_list.append(regress_model)
        
    training_time = datetime.now() - start_time
    oof = pd.concat(oof_list, ignore_index=True)
    if hyper_params_tuning == False:
        print("len(oof):", len(oof))
        print("Done in ", training_time) 
        dict_res = {'oof': oof, 'model_list': model_list, 'feature_importance_lgbm': feature_importance_lgbm, 
                'feature_cols': feature_cols, 'cate_cols': cate_cols, 'X_train': X_train}
        
        if default_params['objective'] == "rmse":
            print("rmse in all the 5 folds:", np.sqrt(mean_squared_error(oof.pred, oof.label))) 
        elif default_params['objective'] == "mse":
            print("mse in all the 5 folds:", mean_squared_error(oof.pred, oof.label) ) 
            
        return dict_res
    else:
        if default_params['objective'] == "rmse":
            return np.sqrt(mean_squared_error(oof.pred, oof.label)), default_params
        if default_params['objective'] == "mse":
            return mean_squared_error(oof.pred, oof.label), default_params
        else:
            print("Need to check the metric")
            return None