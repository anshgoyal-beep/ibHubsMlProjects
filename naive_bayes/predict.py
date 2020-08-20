import numpy as np
import csv
import sys
import pickle

import pandas as pd
import nltk
nltk.download('stopwords')
"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text

def remove_spl_chars_except_space(s): 
    i = 0
    s_with_no_spl_chars = ""
    #using ASCII Values of characters
    for i in range(len(s)): 
        if (ord(s[i]) >= ord('A') and
            ord(s[i]) <= ord('Z') or 
            ord(s[i]) >= ord('a') and 
            ord(s[i]) <= ord('z') or
            ord(s[i]) == ord(' ')):
            s_with_no_spl_chars += s[i] 
            
    return s_with_no_spl_chars 


def preprocessing(s):
    s = remove_spl_chars_except_space(s)
    s =' '.join(s.split()) # replaces multiple spaces with single space
    s = s.lower() #convert to lowercase
    return s


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = pd.read_csv(test_X_file_path,sep = '\t',header=None,names=['text'])
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    test_X['text'].apply(process)
    Y = model.predict(test_X['text'])
    Y= Y.tolist()
    Y=[Y]
    Y=np.asarray(Y)
    Y=(Y.T).tolist()

    return Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
