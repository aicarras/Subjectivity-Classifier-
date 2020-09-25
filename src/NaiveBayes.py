from Model import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import lil_matrix
import scipy as sp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class NaiveBayes(Model):

    def __init__(self,train_data, test_data):

        self.model = GaussianNB()
        self.data = train_data
        self.test = test_data
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()


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
        X = X.toarray()
        Y = self.data[1]

        print("\nTraining naive bayes classifier...\n")
        self.model.fit(X,Y)


    def predict (self, custom_input = None):

        self.train()

        X_T = self.bag_of_words(self.test, "test")
        X_T = X_T.toarray()
        Y_T = self.test[1]
    
        print("Testing naive bayes classifier...\n")

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
        user_input = self.bag_of_words(custom_input, "test")
        user_input = user_input.toarray()
        
        print(custom_input[0], "was predicted as {}".format(self.model. predict(user_input)))