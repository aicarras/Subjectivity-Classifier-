from Model import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
import pickle


class SVM (Model):


    def __init__(self,train_data, test_data, hyperopt, pretrained_file_name = None):   
        self.model = svm.SVC() #svm.SVC(kernel = 'sigmoid') 
        self.opt_model = None
        self.data = train_data
        self.test = test_data
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()
        self.pretrained_file_name = pretrained_file_name
        self.hyperopt = hyperopt


    def bag_of_words(self, data, train_test):
    

        if train_test == "train":

            bag_of_words = self.count_vect.fit(data[0])
            bag_of_words = self.count_vect.transform(data[0])
            X_train_tfidf = self.tfidf_transformer.fit_transform(bag_of_words)
            X_train_tfidf.shape

        elif train_test == "test":

            bag_of_words = self.count_vect.transform(data[0])
            X_train_tfidf = self.tfidf_transformer.transform(bag_of_words)
            X_train_tfidf.shape

        return X_train_tfidf


    def train (self):   
    
        X = self.bag_of_words(self.data, "train")
        Y = self.data[1]

        X = X.toarray()

        print("\nTraining SVM...\n")

        if self.hyperopt:

            #parameter_grid = {'kernel' : ['linear', 'rbf'],  'gamma': [1e-3, 1e-4],'C' : [1, 10]}
            parameter_grid = {'gamma': [1e-3, 1e-4],'C' : [1, 10]}
            self.opt_model = GridSearchCV(svm.SVC(), parameter_grid)

            self.opt_model.fit(X,Y)

        else:

            if self.pretrained_file_name == None:
                self.model.fit(X,Y)
            


    def predict (self,custom_input = None):
    
        self.train()

        if self.pretrained_file_name != None:

            self.model = load(self.pretrained_file_name)
        


        X_T = self.bag_of_words(self.test, "test")
        X_T = X_T.toarray()
        Y_T = self.test[1]
       


        if self.hyperopt:

            print("Testing optimised SVM...\n")

            print("best parameters: ")
            print(self.opt_model.best_params_)

            predicted = self.opt_model.predict(X_T)
            print("Accuracy of optimised SVM: ", np.mean(predicted == Y_T))

            #Save model
            #dump(self.model, 'SVM1000POSGrid.joblib')

        else:

            print("Testing SVM...\n")
            predicted = self.model.predict(X_T)
            print("Accuracy of: ", np.mean(predicted == Y_T))

            #In binary classification, the count of true negatives is C=0-->0, false negatives is C=0-->1, true positives is C=1 -->1 and false positives is C=1 -->0
            tn, fp, fn, tp = confusion_matrix(predicted,Y_T).ravel()
            print(confusion_matrix(predicted,Y_T))
            print("tn:",tn)
            print("fp:",fp)
            print("fn:",fn)
            print("tp:",tp)

            #doc = [["The choice of Miers was praised by the Senate’s top Democrat, Harry Reid of Nevada"],[0]]
            if custom_input != None:
                user_input = self.bag_of_words(custom_input, "test")
                user_input = user_input.toarray()
                print(custom_input[0], "was predicted as {}".format(self.model. predict(user_input)))

            #Save model
            #dump(self.model, 'SVM10000POS.joblib')
        