# 1. Read Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "data/"
origional_data = pd.read_csv(data_dir + "spam.csv", encoding = "latin-1")
print(origional_data.head())
print(origional_data.shape)
label = origional_data.v1
data = origional_data.v2
print(label.head())
print(data.head())

############################################################
# 2. Train using model of SKLearn
# 80% of data for traning dataset and 20% of data for test dataset
from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(data, 
                                                                  label, 
                                                                  test_size = 0.2, 
                                                                  random_state = 0)

print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)
print(data_train.head())
print(data_test.head())

# Build class of vectorizer
class Vectorizer(object):
    def __init__(self):
        pass
        
    def fit_transform(self, data):
        vocab_set = set([])
        for line in data:
            words = line.split()
            for word in words:
                vocab_set.add(word.lower())
        self.vocablary = list(vocab_set)
        
        word_matrix = []
        for line in data:
            words = line.split()
            word_array = np.zeros(len(self.vocablary))
            for word in words:
                if word.lower() in self.vocablary:
                    word_array[self.vocablary.index(word.lower())] += 1
            word_matrix.append(word_array)
        return word_matrix
    
    def transform(self, data, vocablary):
        word_matrix = []
        for line in data:
            words = line.split()
            word_array = np.zeros(len(vocablary))
            for word in words:
                if word.lower() in vocablary:
                    word_array[vocablary.index(word.lower())] += 1
            word_matrix.append(word_array)
        return word_matrix

vectorizer = Vectorizer()
data_train_counts = vectorizer.fit_transform(data_train)
data_test_counts = vectorizer.transform(data_test, vectorizer.vocablary)
print ('Number of all the unique words : ' + str(len(vectorizer.vocablary)))

print("train:")
print(len(data_train_counts))

print("test:")
print(len(data_test_counts))

word_freq = pd.DataFrame({"word": vectorizer.vocablary, "occurrences": np.array(data_train_counts).sum(axis = 0)})
word_freq["frequency"] = word_freq.occurrences / np.sum(word_freq.occurrences)

plt.plot(word_freq.occurrences)
plt.xlabel("index")
plt.ylabel("word occurrences")
plt.show()

word_freq_sort = word_freq.sort_values(by = "occurrences", ascending = False)
word_freq_sort.head()

# Build class of naive bayes
class NBayes(object):
    def __init__(self):
        pass
    
    def fit(self, word_matrix, label):
        num_docs = len(word_matrix)
        num_words = len(word_matrix[0])
        
        # Laplace smoothing
        spam_vector_count = np.ones(num_words);
        ham_vector_count = np.ones(num_words)  
        spam_total_count = num_words;
        ham_total_count = num_words                  
    
        spam_count = 0
        ham_count = 0
        
        for i in range(num_docs):
            if i % 500 == 0:
                print ('Train on the doc id:' + str(i))
            
            if label[i] == 'spam':
                ham_vector_count += word_matrix[i]
                ham_total_count += np.sum(word_matrix[i])
                ham_count += 1
            else:
                spam_vector_count += word_matrix[i]
                spam_total_count += np.sum(word_matrix[i])
                spam_count += 1
        print (ham_count)
        print (spam_count)
    
        self.p_spam_vector = np.log(ham_vector_count/ham_total_count)#注意
        self.p_spam = np.log(spam_count/num_docs)
        self.p_ham_vector = np.log(spam_vector_count/spam_total_count)#注意
        self.p_ham = np.log(ham_count/num_docs)
        #return p_spam_vector, np.log(spam_count/num_docs), p_ham_vector, np.log(ham_count/num_docs)
    
    def predict(self, test_matrix):
        predictions = []
        for test_vector in test_matrix:
            spam = np.sum(test_vector * self.p_spam_vector) + self.p_spam
            ham = np.sum(test_vector * self.p_ham_vector) + self.p_ham
            if spam > ham:
                predictions.append("spam")
            else:
                predictions.append("ham")
        return predictions
  
# Traning begin
bayes = NBayes()
bayes.fit(data_train_counts, label_train.values)
label_pred = bayes.predict(data_test_counts)
print(len(label_pred))

#######################################################
# 3. How is this model: validation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cross_validation import cross_val_score

print("accuracy: ", accuracy_score(label_test, label_pred))
print(classification_report(label_test, label_pred))
print(confusion_matrix(label_test, label_pred))
