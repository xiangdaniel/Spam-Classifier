# Build the class of naive bayes
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
    
        self.p_spam_vector = np.log(ham_vector_count/ham_total_count)
        self.p_spam = np.log(spam_count/num_docs)
        self.p_ham_vector = np.log(spam_vector_count/spam_total_count)
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
