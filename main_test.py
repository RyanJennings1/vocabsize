import random

import numpy as np
import pandas as pd

from unittest import TestCase

from main import *

random.seed(0)

class MainTest(TestCase):
  def test_function1_squares_result(self):
    a = 2
    a_sq = 4
    self.assertEqual(a_sq, function1(a))

  def test_likelihood_returns_expected_result(self):
    M = 2
    ps = np.array([0.1, 0.2, 0.4])
    pu = np.array([0.1, 0.2, 0.4])
    want = 0.008169652224

    given = likelihood(M, ps, pu)

    self.assertAlmostEqual(want, given)

  def test_max_likelihood_returns_expected_result(self):
    ps = np.array([0.1, 0.2, 0.4])
    pu = np.array([0.1, 0.2, 0.4])
    want_index = 0

    given_index = max_likelihood(ps, pu)

    self.assertEqual(want_index, given_index)

  def test_scipy_max_likelihood_returns_expected_result(self):
    ps = np.array([0.1, 0.2, 0.4])
    pu = np.array([0.1, 0.2, 0.4])
    want = 2.5145263671875

    given = scipy_max_likelihood(ps, pu)

    self.assertAlmostEqual(want, given)

  def test_estimate_words_seen_returns_expected_result(self):
    corpus_path: str = 'test_corpus.csv'
    vocab_est = VocabSizeEstimator(corpus_path)
    sample = pd.DataFrame([0.1, 0.2, 0.2, 0.4], columns=['freq'])
    known = [False, True, False, True]
    want = 3

    given = vocab_est.estimate_words_seen(sample, known)

    self.assertEqual(want, given)

  def test_estimate_words_known_corpus_returns_expected_result(self):
    corpus_path: str = 'test_corpus.csv'
    vocab_est = VocabSizeEstimator(corpus_path)
    words_known = 5
    want = 0

    given = vocab_est.estimate_words_known_corpus(words_known)

    self.assertEqual(want, given)

  def test_generate_sample_returns_expected_result(self):
    corpus_path: str = 'test_corpus.csv'
    vocab_est = VocabSizeEstimator(corpus_path)
    sample_size = 5
    rare_threshold = 1e-3
    want = [
      0.0004405917044804,
      0.0003782841492232,
      0.0007504943350108,
      0.0005060370787391,
      0.0004304921585072,
    ]

    given = vocab_est.generate_sample(sample_size, rare_threshold)

    self.assertEqual(want, given['freq'].tolist())

  def test_generate_sample_with_indices_returns_expected_result(self):
    corpus_path: str = 'test_corpus.csv'
    vocab_est = VocabSizeEstimator(corpus_path)
    sample_size = 10
    want_sample = [
      (1, 'I'),
      (3, 'she'),
      (5, 'one'),
      (8, 'more'),
      (10, 'woman'),
      (11, 'should'),
      (15, 'same'),
      (16, 'problem'),
      (19, 'government'),
    ]
    want_freqs = [
      0.0078662927672916,
      0.0029364271740978,
      0.0015189495697124,
      0.0010232730275629,
      0.0006750601380707,
      0.0006134564665971,
      0.0004405917044804,
      0.0004304921585072,
      0.0003782841492232,
    ]

    given_sample, given_freqs = vocab_est.generate_sample_with_indices(sample_size)

    self.assertEqual(want_sample, list(given_sample))
    self.assertEqual(want_freqs, given_freqs['freq'].tolist())