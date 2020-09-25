from Model import Model
import keras
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class NeuralNetwork(Model):


    def __init__(self, train_data, dev_data, test_data, hyperopt, pretrain_data_embedding = None):
            #super(print("Using Naïve Bayes classification"))
        self.model = Sequential()
        self.train_data = train_data
        self.dev = dev_data
        self.test = test_data
        self.embedding = pretrain_data_embedding
        self.vocabulary = None
        self.tk = None
        self.hyperopt = hyperopt

        #Parameters
        self.output_dim = 100
        self.kernel_size = 5
        self.dropout_rate = 0.3
        self.units = 8


    def form_vocabulary (self,data):
    
        #Custom voacabulary for keras embedding
        vocab_prep = [sentence.strip().split() for sentence in data[0]]
        vocab_prep = [word for sentence in vocab_prep for word in sentence]
        vocab_prep = set(vocab_prep)
        self.vocabulary = [word for word in vocab_prep]
        self.tk = Tokenizer(num_words=len(self.vocabulary))
        self.tk.fit_on_texts(self.vocabulary)


    def preprocessing_input(self,data):

        data_input = pad_sequences(self.tk.texts_to_sequences(data))   

        return data_input


    def sequential_model(self, output_dim, kernel_size, dropout_rate, units):

        emb = keras.layers.Embedding(input_dim = len(self.vocabulary), output_dim = output_dim)
        self.model.add(emb)
        
        self.model.add(keras.layers.Dropout(dropout_rate))
        
        #self.model.add(keras.layers.LSTM(50, return_sequences=True))
        #self.model.add(keras.layers.Lambda(lambda x: keras.backend.max(x, axis = 1)))
        self.model.add(keras.layers.Conv1D(output_dim, kernel_size, activation='relu'))

        #self.model.add(keras.layers.GlobalMaxPooling1D()) #Flatten into one dimensional vector
        self.model.add(keras.layers.GlobalAveragePooling1D())
        
        #self.model.add(keras.layers.Dense(units,  activation= keras.activations.tanh))
        self.model.add(keras.layers.Dense(units,  activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='sigmoid')) #Pull apart results in a value between 0-1

        print("Model summary:")
        self.model.summary()

        self.model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

        return self.model

    def random_search_optimisation(self):

        parameter_grid = dict(output_dim = [50, 100, 300], epochs = [20,50,100], kernel_size = [3,5,10], dropout_rate = [0.1,0.2,0.3], units = [10, 25, 50])
        opt_model = KerasClassifier(build_fn= self.sequential_model, batch_size=15) 
        grid_space = RandomizedSearchCV(estimator=opt_model, param_distributions= parameter_grid, n_iter=5)
        grid_result = grid_space.fit(self.preprocessing_input(self.train_data[0]), self.train_data[1])
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # Evaluate testing set
        test_accuracy = grid_space.score(self.preprocessing_input(self.test[0]), self.test[1])
        print("Random search optimisation NN model ACCURACY: ", test_accuracy)


    def train(self):

        print('\nTraining Neural Network...\n')

        self.form_vocabulary(self.train_data[0])

        if self.hyperopt:

            self.random_search_optimisation()

        else:   

            self.sequential_model(self.output_dim,self.kernel_size,self.dropout_rate,self.units)
            self.model.fit(self.preprocessing_input(self.train_data[0]), self.train_data[1], 
                        batch_size=32, 
                        epochs=30, 
                        validation_data = (self.preprocessing_input(self.dev[0]),self.dev[1]))


    def predict(self, custom_input = None):

        if self.hyperopt:
            print("\nRandomSearch optimisation: \n")
            self.train()

        else:

            self.train()

            print('Evaluation')
            loss, acc = self.model.evaluate(self.preprocessing_input(self.test[0]), self.test[1], batch_size=32)
            print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
            # if custom_input != None:
            #     #user_input = self.bag_of_words(custom_input, "test")
            #     #user_input = user_input.toarray()
            #     print(custom_input[0], "was predicted as {}".format(self.model.predict(np.array(custom_input[0]))))
