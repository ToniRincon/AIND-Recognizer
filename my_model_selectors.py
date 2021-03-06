import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # raise NotImplementedError
        
        # best score is the s
        best_bic=float('inf')
        best_model=None
        
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)
                logL= model.score(self.X,self.lengths)
                n_features = len(self.X[0])
                p = n_components**2+2*n_components*n_features-1
                N = len(self.X)
                bic = -2*logL+p*np.log(N)
                if bic<best_bic:
                    best_bic = bic
                    best_model = model
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # raise NotImplementedError
        
        M = len(self.hwords)
        best_dic=float('-inf')
        best_model=None
        
        for n_components in range(self.min_n_components, self.max_n_components+1):
            #print(self.this_word,n_components)
            try:
                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)
                #print(self.this_word,n_components, 'model')
                logL= model.score(self.X,self.lengths)
                #print(self.this_word,n_components,'score')
                other_logL_sum=0
                for other_word in self.words:
                    if other_word == self.this_word:
                        continue
                    #print(self.this_word,n_components,other_word)
                    other_X, other_lengths = self.hwords[other_word]
                    other_logL = model.score(other_X,other_lengths)
                    #print(self.this_word,n_components,other_word,'score')
                    other_logL_sum += other_logL
                dic = logL-other_logL_sum/(M-1)
                #print(self.this_word,n_components,'dic',dic)
                
                if dic>best_dic:
                    best_dic = dic
                    best_model = model
                    #print(self.this_word,n_components,'best')
            except:
                #print(self.this_word,n_components,'except')
                pass
        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        best_cv=float('-inf')
        best_model=None
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                split_n = min(len(self.sequences),3) # in case there are least than 3 sequences
                split_method = KFold(split_n)
                logL_sum = 0
                for train_index, test_index in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(train_index, self.sequences)
                    X_test, lengths_test  = combine_sequences(test_index, self.sequences)
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(X_train, lengths_train)
                    logL = model.score(X_test,lengths_test)
                    logL_sum += logL
                logL_avg = logL_sum/split_n
                if logL_avg>best_cv:
                    best_cv = logL_avg
                    best_model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)
            except:
                pass
        return best_model
