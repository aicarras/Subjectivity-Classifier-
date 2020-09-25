import sys, argparse
import re 



def main():

    file_name = "../data/biased.full" #"../data/neutral"
    target_file = "../data/target_biased.txt"  #"../data/test.txt"

    read_write(file_name,target_file)


def read_write(file_name,target_file):

    text = reader(file_name)
    written = writer(text,target_file)

    return written


def reader(file_name):

    with open(file_name,encoding="utf-8") as f:
        
        text = []
        all_text = [l.strip() for l in f.readlines()]

        for example in all_text:

            if example[:7].isnumeric:
                target = example[10:25]
                #Method for finding substring retrieved from: https://www.geeksforgeeks.org/python-ways-to-find-nth-occurrence-of-substring-in-a-string/
                end_target = -1
                for i in range(0,2):
                    end_target = example.find(target,end_target+1)
                #OR use regex
                #reps = [repetition.start() for repetition in re.finditer(target, example)]
                #end_target = reps[1] #returns the second occurence of the target sentence
            sentence = example[7:end_target]

            counter = 0
            while (sentence[counter].isnumeric()):

                counter += 1
            
            sentence = sentence[counter:]
            text.append(sentence.strip())

    return text

def writer(sentences, target_file):

    with open(target_file, "w") as f:
        for sentence in sentences:
            f.write(sentence)
            f.write("\n")

    return True



if __name__ == "__main__":
   main()