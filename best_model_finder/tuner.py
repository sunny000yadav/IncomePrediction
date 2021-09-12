from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

class Model_Finder:

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.gnb = GaussianNB()
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)

    def get_best_params_for_naive_bayes(self,train_x,train_y):

        self.logger_object.log(self.file_object,'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:

            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            self.grid = GridSearchCV(estimator=self.gnb, param_grid=self.param_grid, cv=5,  verbose=3)

            self.grid.fit(train_x, train_y)

            self.var_smoothing = self.grid.best_params_['var_smoothing']

            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)

            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:

            self.param_grid_xgboost = {

                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],"max_depth": range(8, 10, 1)

            }

            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)

            self.grid.fit(train_x, train_y)
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.xgb = XGBClassifier(criterion=self.criterion, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )

            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):

        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')

        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)

            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))

            self.naive_bayes=self.get_best_params_for_naive_bayes(train_x,train_y)
            self.prediction_naive_bayes=self.naive_bayes.predict(test_x)

            if len(test_y.unique()) == 1:
                self.naive_bayes_score = accuracy_score(test_y,self.prediction_naive_bayes)
                self.logger_object.log(self.file_object, 'Accuracy for NB:' + str(self.naive_bayes_score))
            else:
                self.naive_bayes_score = roc_auc_score(test_y, self.prediction_naive_bayes) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.naive_bayes_score))


            if(self.naive_bayes_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'NaiveBayes',self.naive_bayes

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_model method of the Model_Finder class. Exception message:  '
                                   + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

