import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import optimize

# helper functions

def function1(x):
  return x**2

def likelihood(M, ps, pu):
  '''Computes the likelihood that you have seen M words given arrays 
  representing words you do and don't know. ps, pu are np arrays of the 
  frequencies of the words you do and don't know, respectively.'''
    
  return np.prod(1-(1-ps)**M)*np.prod((1-pu)**M)

def max_likelihood(ps, pu):
  '''Computes the maximum likelihood value of M = number of words seen
  givenn ps, pu arrays.

  Currently just brute forcing it.'''
    
  return np.argmax(np.array([likelihood(m, ps, pu) for m in range(0, 50000000, 10000)]))

def scipy_max_likelihood(ps, pu):
  '''Compute ML estimate of M using scipy optimization function.'''

  f = lambda x: -likelihood(x,ps,pu)

  init_guess = 1 / (np.min(ps))

  return optimize.fmin(f,init_guess,disp=False)[0]


class VocabSizeEstimator:
  '''Class to extimate your vocab size based on words you do and don't know and 
  their frequency in a self.corpus.'''

  def __init__(self,corpus_path):
    '''self.Corpus should be a csv file with a list of word, relative frequency.'''

    print('Initializing...')

    #load in corpus
    self.corpus = pd.read_csv(corpus_path, header=None)
    self.corpus.columns = ['word','freq']

    #calculate unique word conversion from words seen
    #need to approximate to get to big enough numbers
    self.probs = np.array(self.corpus['freq'])


  def estimate_words_seen(self,sample,known):
    '''Get estimate of total words seen, given a sample of words (dataframe) and
    a boolean array of whether you know them.'''

    sample_p = list(sample['freq'])
    s = np.array([x for x,y in zip(sample_p,known) if y])
    u = np.array([x for x,y in zip(sample_p,known) if not y])

    return int(scipy_max_likelihood(s,u))

  def estimate_words_known_corpus(self, words_seen):
    '''Now we need to go from words seen to words known, i.e. *unique* words seen.
    Here we base it on the frequencies of words in the self.corpus. Note that this assumes
    there are no other words, so the estimate is bounded by the size of the self.corpus freq table.
    [The derivation of this equation is a little involved]'''

    s = np.vectorize(lambda x: (x-1)*((1-x)**words_seen-1))

    return int(np.sum(s(self.probs)))

  def estimate_words_known_theoretical(self,words_seen):
    '''Calculate unique words seen using a theoretical distribution of word frequencies.
    Takes parameters A and alpha describing the frequency ditribution:
    f(i) = A/i^alpha where i is the rank of a word in frequency.'''

    pass 

  def generate_sample(self, size, rare_threshold):
    '''Generate a sample of rare words from the self.corpus list.'''

    rare_words = self.corpus[self.corpus['freq'] < rare_threshold]
    rows = random.sample(rare_words.index.tolist(), size)
    return rare_words.loc[rows]

  def generate_sample_with_indices(self, size):
    """Generate a sample of words from the corpus with thier indices."""
    rare_words = self.corpus
    bins = np.linspace(0, len(rare_words.index.tolist()), size)
    rows = []
    for i in range(len(bins)-1):
      words_slice = rare_words.index.tolist()[round(bins[i]):round(bins[i+1])]
      rows += random.sample(words_slice, 1)
    locs = list(rare_words['word'].iloc[rows])
    return zip(rows, locs), rare_words.loc[rows]

  def get_vocab_size(self,sample,known):
    '''Get vocab size estimate based on knowledge of sample of words.'''
    return self.estimate_words_known_corpus(self.estimate_words_seen(sample,known))

  def get_vocab_size_polynomial(self, known):
    """
    Get vocab size estimate based on knowleged of sample of words
    and calculate using an order 1 polynomial function
    """
    corpus = list(np.array(self.corpus['freq']))
    y = [corpus[i] for i in known]
    # get line of best fit through points
    coeffs = np.polyfit(known, y, 1)
    polynomial = np.poly1d(coeffs)
    return round(polynomial.roots[0])

  def run(self):
    sample = self.generate_sample(20, 1e-7)
    K = []
    for i, word in enumerate(sample['word']):
      print(word)
      while True:
        know = input('Do you know this word y/n? ')
        if know == 'y':
          K.append(1)
          break
        elif know == 'n':
          K.append(0)
          break
        else:
          continue

    print('OK, estimateing vocabulary size...')
    print('Your vocabulary size is ', self.get_vocab_size(sample, K))

  def run_polynomial(self):
    sample, sample2 = self.generate_sample_with_indices(20)
    known_words = []
    K = []
    for s in sample:
      print(s[1])
      while True:
        know = input('Do you know this word y/n? ')
        if know == 'y':
          known_words.append(s[0])
          K.append(1)
          break
        elif know == 'n':
          K.append(0)
          break
        else:
          continue

    print('OK, estimateing vocabulary size...')
    print('Your vocabulary size is (polynomial) ', self.get_vocab_size_polynomial(known_words))
    print('Your vocabulary size is (maximum likelihood) ', self.get_vocab_size(sample2, K))


def pattern1():
  """run pattern 1."""
  s = np.array([0.000001,0.001,0.0004,0.00001])
  u = np.array([0.001])

  print(max_likelihood(s,u))
  print()
  print(scipy_max_likelihood(s,u))

def pattern2():
  """run pattern 2."""
  corpus_path: str = 'corpus_corrected.csv'
  vocab_est = VocabSizeEstimator(corpus_path)
  vocab_est.run()

def expo_dist_random_best_fit():
  """
  Plot exponential distribution with a
  line of best fit going through random
  points on the curve
  """
  # plot exponential decay
  size = 1000
  y = sorted(np.random.exponential(1, size), reverse=True)
  plt.plot(y, 'r')
  # get 10 random points on exponential
  ran_x, ran_y = zip(*random.sample(list(enumerate(y)), 10))
  # get line of best fit through points
  coeffs = np.polyfit(ran_x, ran_y, 1)
  polynomial = np.poly1d(coeffs)
  print(f'polynomial at 0 = {polynomial(0)}')
  print(f'polynomial roots = {polynomial.roots}')
  x_axis = np.linspace(0, size, size)
  y_axis = polynomial(x_axis)
  # plot line on same graph
  plt.plot(x_axis, y_axis, 'b')
  for i in range(0, len(ran_x)):
    plt.plot(ran_x[i], ran_y[i], 'bo')
  plt.show()

def pattern6():
  """
  Plot frequencies of words which form an
  exponential distribution, then mark 10
  random points and plot a line of best fit
  through them
  """
  # plot exponential decay
  corpus_path: str = 'corpus_corrected.csv'
  corpus = pd.read_csv(corpus_path, header=None)
  corpus.columns = ['word', 'freq']

  y = list(np.array(corpus['freq']))
  plt.plot(y, 'r')

  # get 10 random points on exponential
  ran_x, ran_y = zip(*random.sample(list(enumerate(y)), 10))
  # get line of best fit through points
  coeffs = np.polyfit(ran_x, ran_y, 1)
  polynomial = np.poly1d(coeffs)
  print(f'polynomial at 0 = {polynomial(0)}')
  print(f'polynomial roots = {polynomial.roots}')
  x_axis = np.linspace(0, len(y), len(y))
  y_axis = polynomial(x_axis)
  # plot line on same graph
  plt.plot(x_axis, y_axis, 'b')
  for i in range(0, len(ran_x)):
    plt.plot(ran_x[i], ran_y[i], 'bo')
  plt.show()

def vocab_est_with_poly_func():
  """
  Run vocabulary estimator and calculate
  vocabulary size based on polynomial
  function
  """
  corpus_path: str = 'corpus_corrected.csv'
  vocab_est = VocabSizeEstimator(corpus_path)
  vocab_est.run_polynomial()

if __name__=='__main__':
  vocab_est_with_poly_func()
