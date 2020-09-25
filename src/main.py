import sys, argparse, random
import nltk
from NaiveBayes import NaiveBayes
from SVM import SVM
from NN import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description='Script to train a language model')
    parser.add_argument("--training_size", default= 10000, type=int, help="define training data set size")
    parser.add_argument("--test_size", default= 1000, type=int, help="define test data set size")
    parser.add_argument("--data_biased", default="../data/target_biased.txt", type=str, help="text file containing the source biased data (all)")
    parser.add_argument("--data_neutral", default="../data/target_neutral.txt", type=str, help="text file containing the source neutral data")
    parser.add_argument("--model", default = "NB", type = str, help = "choose model ||| NB --> Naïve Bayes ||| SVM --> Support Vector Machine ||| NN --> Neural Network")
    parser.add_argument("--user_input", default=False, type=bool, help="Command line input for custom sentence classification")
    parser.add_argument("--load_model", default=False, type=bool, help="Load the best performing SVM model")
    args = parser.parse_args()

    #Load data from files
    biased_data = load_data(args.data_biased, "biased")
    neutral_data = load_data(args.data_neutral, "neutral")

    #Split data into sets
    training_data, dev_data, test_data = prepare_data(args.training_size, args.test_size, biased_data, neutral_data)

    #Get POS tags
    training_data_pos = pos_tag(training_data)
    dev_data_pos = pos_tag(dev_data)
    test_data_pos = pos_tag(test_data)

    print("Length of TRAINING data: ",len(training_data[0]))
    print("Length of DEVELOPMENT data: ",len(dev_data[0]))
    print("Length of TEST data: ",len(test_data[0]))


    if (args.model == "NB"):

        model = NaiveBayes(training_data_pos,test_data_pos)

    elif (args.model == "SVM"):

        if args.load_model:

            model = SVM(training_data_pos, test_data_pos, False,  "SVM10000POS.joblib")
        else:

             model = SVM(training_data_pos, test_data_pos, False)

    elif (args.model == "NN"):

        model  = NeuralNetwork(training_data_pos, dev_data_pos, test_data_pos, False) 


    if args.user_input:
        user_input = input("Enter the sentence that you want to classify: ")
        user_label = input("Do you think its biased or not? Enter 1 (for biased) | 0 (for neutral): ")
        user_data = [[user_input],[user_label]]
        model.predict(pos_tag(user_data))
    else:

        model.predict()


def pos_tag(data):

    sentences = data[0]
    labels = data[1]
    tagged_sentences = []
    for sentence in sentences:

        tagged = nltk.pos_tag(nltk.word_tokenize(sentence)) 
        sent = ' '
        for t in tagged:
            tag = t[0] + '/' + t[1]
            sent += ''.join(tag)
            sent += ''.join(' ')
        
        tagged_sentences.append(sent.strip())

    
    output = [tagged_sentences, labels]

    return output

def load_data(file_name, ident):

    data = []
    labels = []
    label = None 

    if ident == "biased":
        label = 1
    elif ident == "neutral":
        label = 0
     
    with open(file_name,encoding="utf-8") as f:
        for l in f.readlines():
            data.append(l.strip())#.split())
            labels.append(label)

    return [data,labels]

def shuffle(data):

    sentences = data[0]
    labels = data[1]

    #Implememntation inspired from: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    zipped = list(zip(sentences,labels))
    random.shuffle(zipped)
    #Implementation inspired from: https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
    shuffled = [list(t) for t in zip(*zipped)] 

    return shuffled
    
def prepare_data(train_size, test_size, biased_data, neutral_data):

    train_size = int(train_size/2)
    test_size = int(test_size/2)
    distrb_size = train_size#int(train_size/2)
    train_slice = slice(0,distrb_size)
    alloc_size = distrb_size + 2500
    dev_size = slice(distrb_size,alloc_size)
    end = alloc_size + test_size
    test_slice = slice(alloc_size, end)

    biased_training_sets = biased_data[0][train_slice]
    biased_training_labels = biased_data[1][train_slice]
    neutral_training_sets = neutral_data[0][train_slice]
    neutral_training_labels = neutral_data[1][train_slice]

    training_data = [biased_training_sets + neutral_training_sets, biased_training_labels + neutral_training_labels]
    training_data = shuffle(training_data)

    biased_dev_sets = biased_data[0][dev_size]
    biased_dev_labels = biased_data[1][dev_size]
    neutral_dev_sets = neutral_data[0][dev_size]
    neutral_dev_labels = neutral_data[1][dev_size]

    dev_data = [biased_dev_sets + neutral_dev_sets, biased_dev_labels + neutral_dev_labels]
    dev_data = shuffle(dev_data)

    biased_test_sets = biased_data[0][test_slice]
    biased_test_labels = biased_data[1][test_slice]
    neutral_test_sets = neutral_data[0][test_slice]
    neutral_test_labels = neutral_data[1][test_slice]

    test_data = [biased_test_sets + neutral_test_sets, biased_test_labels + neutral_test_labels]
    test_data = shuffle(test_data)

    return  training_data, dev_data, test_data

if __name__ == "__main__":
   main()