
# random forest class for tuning

class rf_model():


        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'n_estimators']:
            params[parameter_name] = int(params[parameter_name])
                
        rf_results = RandomForestClassifier(**params, random_state=randomseed)
        #rf_results.fit(X_train, y_train)
        rf_cv_scores = sklearn.model_selection.cross_val_predict(rf_results, X_train, y_train, cv=5, verbose=False)        
        recall_score = sklearn.metrics.recall_score(y_pred=rf_cv_scores, y_true=y_train)
        precision_score = sklearn.metrics.precision_score(y_pred=rf_cv_scores, y_true=y_train)
        f1_score = sklearn.metrics.f1_score(y_pred=rf_cv_scores, y_true=y_train)

        return {'loss': (1 - recall_score), 'status': STATUS_OK, 'params': params, 'iteration': ITERATION}
    
        space = {
            'max_depth' : hp.quniform('max_depth', 5, 10, 1),
            'max_features': hp.choice('max_features', range(20, int((X_train.shape[:][1])/5))),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'n_estimators': hp.choice('n_estimators', np.arange(200, 1000))
        }
        
        # Run optimization
        best = fmin(fn = rf_model.rf_score, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = trials, rstate = np.random.RandomState(randomseed))
        best = trials.best_trial['result']['params']
        return best, trials
    
    def rf_train(best_params):
        model = RandomForestClassifier(random_state = randomseed)
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        
    def rf_cv(X_train, y_train, best):
        model = RandomForestClassifier(**best, verbose=False)
        rf_cv_scores = sklearn.model_selection.cross_val_predict(model, X_train, y_train, cv=5)
