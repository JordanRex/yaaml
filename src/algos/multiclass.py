import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, RidgeClassifierCV, PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# multi-class classification/regression superclass

class opr_model():
    
    """ Algorithms in each function are as follows:
    1. opr_multiclass_inherant
    2. opr_multiclass_oneVS
    3. opr_regression
    4. opr_ordinal
    5. opr_neuralnets
    6. ...
    
    P.S: Not good enough algos commented
    """
    
    def __init__(self, train_df, valid_df, ytrain_vec, yvalid_vec, algo=0):
        # initialize the arguments
        self.train = train_df
        self.valid = valid_df
        self.ytrain = ytrain_vec
        self.yvalid = yvalid_vec
        self.algo = algo
        
        # print stats
        print(train_df.shape, '\n', valid_df.shape, '\n', ytrain_vec.shape, '\n', yvalid_vec.shape)
        print('the class values are:', yvalid_vec.unique(), '\n')
        
        # run the various algorithm functions
        self.opr_multiclass_inherant()
        self.opr_neuralnets()
        self.opr_multiclass_oneVS()
        self.main()
        self.ensemble()
        
    def main(self):
        names = ['Logit', 'AdaBoost', 'ExtraTrees', 'LinearSVC', 'Ridge',
                'GBM', 'OneVsRest', 'OneVsOne', 'OutputCode']
        classifiers = [self.logr, self.adab, self.extt, self.lsvc, self.rdgc,
                      self.rfcf, self.gbmc, self.ovrc, self.ovoc, self.occf]
        
        #testnames = ['Logit', 'ExtraTrees', 'MLP']
        #testclassifiers = [self.logr, self.extt, self.mlpc]
        
        # iterate over classifiers
        clf_scores = {}
        clf_probs = {}
        clf_predictions = {}
        clf_insample = {}
        for name, clf in zip(names, classifiers):
            print(name, 'is happening \n')
            clf.fit(self.train.values, self.ytrain)
            clf_scores[name] = clf.score(self.valid.values, self.yvalid)
            print(clf_scores[name], '\n')
            # predict probs
            if hasattr(clf, "predict_proba"):
                clf_probs[name] = clf.predict_proba(self.valid.values)
                clf_insample[name] = clf.predict_proba(self.train.values)
            else:
                clf_probs[name] = clf.predict(self.valid.values)
                clf_insample[name] = clf.predict(self.train.values)
            # predictions as well
            clf_predictions[name] = clf.predict(self.valid.values)
        
        self.scores = clf_scores
        self.probs = clf_probs
        self.predictions = clf_predictions
        self.insample = clf_insample
        return None
        
    def opr_multiclass_inherant(self):
        # all sklearn native algorithms
        self.logr = LogisticRegressionCV(random_state=1, multi_class='ovr', max_iter=1000, penalty='l2') #has probs
        self.adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=12, presort=True), 
                                       n_estimators=100, learning_rate=0.1) #has probs
        self.extt = ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1) #has probs
        #self.knnc = KNeighborsClassifier(n_neighbors=3, weights='distance') #has probs
        #self.ldac = LinearDiscriminantAnalysis() #has probs
        #self.qdac = QuadraticDiscriminantAnalysis() #has probs
        self.lsvc = LinearSVC(multi_class='ovr', random_state=1) #multiclass='crammer_singer' another setting #no probs
        self.rdgc = RidgeClassifierCV(cv=5) #no probs
        self.rncf = RadiusNeighborsClassifier(n_jobs=-1, radius=2, outlier_label=[2091]) #no probs
        self.rfcf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=1, 
                                       class_weight={0:1, 1:1, 2:1, 3:1, 4:3, 5:3}) #has probs
        #self.nsvc = NuSVC(kernel='linear', nu=0.7, probability=True, class_weight={0:2, 1:1, 2:1, 3:1, 4:2, 5:2}, random_state=1) #has probs
        #self.ksvc = SVC(kernel='poly', probability=True, class_weight={0:1, 1:1, 2:1, 3:1, 4:3, 5:3}, random_state=1) #has probs
        #self.gpcf = GaussianProcessClassifier(random_state=1, multi_class='one_vs_rest', n_jobs=-1) #has probs
        self.gbmc = GradientBoostingClassifier(learning_rate=0.01, max_depth=12, n_estimators=200, subsample=0.8, 
                                           max_features=0.8, random_state=1) #has probs
        self.sgdc = SGDClassifier(loss='log', penalty='elasticnet', max_iter=20, n_jobs=-1, early_stopping=True,
                             class_weight={0:2, 1:2, 2:1, 3:1, 4:3, 5:4}) #loss ='modified_huber' #has probs
        return None
        
    def opr_multiclass_oneVS(self):
        """ best algorithms found from the opr_multiclass_inherant will be used as the starting base estimator for the 
        methods below. a separate tuning framework to find the best base estimator for oneVS methods will be 
        implemented later """
        self.ovrc = OneVsRestClassifier(ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1), n_jobs=-1) #has probs
        self.ovoc = OneVsOneClassifier(xgb.XGBClassifier(learning_rate=0.01, n_estimators=200, colsample_bytree=0.7, subsample=0.7, 
                    scale_pos_weight=2, objective='multi:softmax', max_depth=10, num_class=6)) #no probs
        self.occf = OutputCodeClassifier(ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1),
                                         code_size=5, random_state=1) #no probs
        return None
        
    def opr_regression(self):
        return None
    
    def opr_ordinal(self):
        return None
    
    def opr_neuralnets(self):
        ### mlp classifier ###
        self.mlpc = MLPClassifier(hidden_layer_sizes=(10,))
        return None
    
    def ensemble(self):
        train_output = self.ytrain.copy()
        valid_output = self.yvalid.copy()
        for k,v in self.insample.items():
            df_insample = pd.DataFrame(self.insample[k])
            df_valid = pd.DataFrame(self.probs[k])
            df_insample.columns = [k+str(i) for i in df_insample.columns.values.tolist()]
            df_valid.columns = [k+str(i) for i in df_valid.columns.values.tolist()]
            train_output = pd.concat([train_output, df_insample], axis=1, ignore_index=False)
            valid_output = pd.concat([valid_output, df_valid], axis=1, ignore_index=False)
        
        ens_ytrain = train_output.response.values
        ens_yvalid = valid_output.response.values
        self.ens_train = train_output.drop(['response'], axis=1, inplace=False)
        self.ens_valid = valid_output.drop(['response'], axis=1, inplace=False)
        
        ens_model = ExtraTreesClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
        ens_model.fit(self.ens_train, ens_ytrain)
        print('ensemble score is:', ens_model.score(self.ens_valid, ens_yvalid))
        self.ensmod = ens_model
        return None
    
    
oprmod = opr_model(train, valid, ytrain, yvalid)
