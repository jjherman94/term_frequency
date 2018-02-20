"""
Joshua Herman
CS 6001
Homework 1
"""
# !/usr/bin/env python3

import itertools
import logging
import os
import re
import string
import sys
from collections import Counter

# Needed to run on certain computer but not all, not sure on reason
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import numpy
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from scipy.sparse.csr import csr_matrix
from scipy.stats import linregress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

contractions = {
    "ain't":        "am not",
    "aren't":       "are not",
    "can't":        "cannot",
    "can't've":     "cannot have",
    "'cause":       "because",
    "could've":     "could have",
    "couldn't":     "could not",
    "couldn't've":  "could not have",
    "didn't":       "did not",
    "doesn't":      "does not",
    "don't":        "do not",
    "hadn't":       "had not",
    "hadn't've":    "had not have",
    "hasn't":       "has not",
    "haven't":      "have not",
    "he'd":         "he had",
    "he'd've":      "he would have",
    "he'll":        "he will",
    "he'll've":     "he will have",
    "he's":         "he is",
    "how'd":        "how did",
    "how'd'y":      "how do you",
    "how'll":       "how will",
    "how's":        "how does",
    "i'd":          "i would",
    "i'd've":       "i would have",
    "i'll":         "i will",
    "i'll've":      "i will have",
    "i'm":          "i am",
    "i've":         "i have",
    "isn't":        "is not",
    "it'd":         "it would",
    "it'd've":      "it would have",
    "it'll":        "it will",
    "it'll've":     "it will have",
    "it's":         "it is",
    "let's":        "let us",
    "ma'am":        "madam",
    "mayn't":       "may not",
    "might've":     "might have",
    "mightn't":     "might not",
    "mightn't've":  "might not have",
    "must've":      "must have",
    "mustn't":      "must not",
    "mustn't've":   "must not have",
    "needn't":      "need not",
    "needn't've":   "need not have",
    "o'clock":      "of the clock",
    "oughtn't":     "ought not",
    "oughtn't've":  "ought not have",
    "shan't":       "shall not",
    "sha'n't":      "shall not",
    "shan't've":    "shall not have",
    "she'd":        "she would",
    "she'd've":     "she would have",
    "she'll":       "she will",
    "she'll've":    "she will have",
    "she's":        "she is",
    "should've":    "should have",
    "shouldn't":    "should not",
    "shouldn't've": "should not have",
    "so've":        "so have",
    "so's":         "so is",
    "that'd":       "that had",
    "that'd've":    "that would have",
    "that's":       "that is",
    "there'd":      "there would",
    "there'd've":   "there would have",
    "there's":      "there is",
    "they'd":       "they would",
    "they'd've":    "they would have",
    "they'll":      "they will",
    "they'll've":   "they will have",
    "they're":      "they are",
    "they've":      "they have",
    "to've":        "to have",
    "wasn't":       "was not",
    "we'd":         "we would",
    "we'd've":      "we would have",
    "we'll":        "we will",
    "we'll've":     "we will have",
    "we're":        "we are",
    "we've":        "we have",
    "weren't":      "were not",
    "what'll":      "what will",
    "what'll've":   "what will have",
    "what're":      "what are",
    "what's":       "what is",
    "what've":      "what have",
    "when's":       "when is",
    "when've":      "when have",
    "where'd":      "where did",
    "where's":      "where is",
    "where've":     "where have",
    "who'll":       "who will",
    "who'll've":    "who will have",
    "who's":        "who is",
    "who've":       "who have",
    "why's":        "why is",
    "why've":       "why have",
    "will've":      "will have",
    "won't":        "will not",
    "won't've":     "will not have",
    "would've":     "would have",
    "wouldn't":     "would not",
    "wouldn't've":  "would not have",
    "y'all":        "you all",
    "y'all'd":      "you all would",
    "y'all'd've":   "you all would have",
    "y'all're":     "you all are",
    "y'all've":     "you all have",
    "you'd":        "you would",
    "you'd've":     "you would have",
    "you'll":       "you will",
    "you'll've":    "you will have",
    "you're":       "you are",
    "you've":       "you have"
}


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def expand_contractions(s, contractions_dict=contractions):
    """
    Expands the contractions in s to allow for better term tokenizing
    """

    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, s)


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.tag.pos_tag(tokens)
    clean_tagged = [token_tagged for token_tagged in tagged
                    if token_tagged[1] not in unwanted_tags]
    tokens = nltk.tag.untag(clean_tagged)
    stems = stem_tokens(tokens)
    stems = [stem for stem in stems if stem.isalpha() and len(stem) > 1]
    return stems


def gen_jaccard_similarity(x, y):
    """
    Performs the generalized jaccard similarity
    """
    n = d = 0
    for a, b in zip(x, y):
        n += min(a, b)
        d += max(a, b)
    if d == 0:
        # Return 1.0 since d==0 if and only if both arrays are all zeros
        return 1.0
    return n / d


def delete_row_csr(mat, index):
    """
    Used for deleting rows of a csr matrix
    https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete
    -equivalent-for-sparse-matrices
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[index + 1] - mat.indptr[index]
    if n > 0:
        mat.data[mat.indptr[index]:-n] = mat.data[mat.indptr[index + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[index]:-n] = mat.indices[mat.indptr[index + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[index:-1] = mat.indptr[index + 1:]
    mat.indptr[index:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def create_histogram(term_freq_matrix):
    """
    Creates the histogram

    :param csr_matrix term_freq_matrix: the term frequence matrix
    """
    logger.debug("Begin function")
    # Create histogram
    avg_freq = []
    for feature in range(term_freq_matrix.shape[1]):
        avg_freq.append(term_freq_matrix.getcol(feature).sum() /
                        term_freq_matrix.shape[0])

    avg_freq.sort(reverse=True)
    y_pos = numpy.arange(term_freq_matrix.shape[1])
    freq_values = [freq[1] for freq in avg_freq]

    plt.bar(y_pos, freq_values, align='center', alpha=0.5)
    plt.ylabel('Avg Frequency')
    plt.title('Frequency Histogram')
    plt.savefig('results/histogram.png')
    plt.close()
    logger.debug("End function")


def create_heatmap(euc_sim_matrix, cos_sim_matrix, jac_sim_matrix):
    """
    Creates the heatmap

    :param numpy.ndarray euc_sim_matrix: pairwise similarity matrix
    :param numpy.ndarray cos_sim_matrix: pairwise similarity matrix
    :param numpy.ndarray jac_sim_matrix: pairwise similarity matrix
    """
    logger.debug("Begin function")

    ax = sns.heatmap(euc_sim_matrix, cmap="hot_r")
    ax.set_title("Euclidean Similarities")
    plt.savefig('results/euclidean_heatmap.png')
    plt.close()

    ax = sns.heatmap(cos_sim_matrix, cmap="hot_r")
    ax.set_title("Cosine Similarities")
    plt.savefig('results/cosine_heatmap.png')
    plt.close()

    ax = sns.heatmap(jac_sim_matrix, cmap="hot_r")
    ax.set_title("Generalized Jaccard Similarities")
    plt.savefig('results/jaccard_heatmap.png')
    plt.close()
    logger.debug("End function")


def perform_linear_regression(euc_list, cos_list, jac_list):
    logger.debug("Begin function")
    # Linear Regression
    euc_sim_array = numpy.asarray([item[1] for item in
                                   euc_list]).transpose()
    cos_sim_array = numpy.asarray([item[1] for item in
                                   cos_list]).transpose()
    jac_sim_array = numpy.asarray([item[1] for item in
                                   jac_list]).transpose()

    # cos vs euc
    slope, intercept, r_value, p_value, std_err = linregress(
            [item[1] for item in euc_list],
            [item[1] for item in cos_list])
    logger.info(
            "slope {}, intercept {}, r_value {}, p_value {}, std_err {}".format(
                    slope, intercept, r_value, p_value, std_err))

    # Plot outputs
    plt.scatter(euc_sim_array, cos_sim_array, color='black')
    plt.plot(euc_sim_array, intercept + slope * euc_sim_array, color='blue',
             linewidth=3)
    plt.xlabel('Euclidean')
    plt.ylabel('Cosine')
    plt.title('Euclidean vs Cosine')
    plt.savefig('results/euc_cos_regression.png')
    plt.close()

    # euc vs jaccard
    slope, intercept, r_value, p_value, std_err = linregress(
            [item[1] for item in jac_list],
            [item[1] for item in euc_list])
    logger.info(
            "slope {}, intercept {}, r_value {}, p_value {}, std_err {}".format(
                    slope, intercept, r_value, p_value, std_err))
    # Plot outputs
    plt.scatter(jac_sim_array, euc_sim_array, color='black')
    plt.plot(jac_sim_array, intercept + slope * jac_sim_array, color='blue',
             linewidth=3)
    plt.xlabel('Jaccard')
    plt.ylabel('Euclidean')
    plt.title('Jaccard vs Euclidean')
    plt.savefig('results/euc_jac_regression.png')
    plt.close()

    # cos vs jaccard
    slope, intercept, r_value, p_value, std_err = linregress(
            [item[1] for item in cos_list],
            [item[1] for item in jac_list])
    logger.info(
            "slope {}, intercept {}, r_value {}, p_value {}, std_err {}".format(
                    slope, intercept, r_value, p_value, std_err))

    # Plot outputs
    plt.scatter(cos_sim_array, jac_sim_array, color='black')
    plt.plot(cos_sim_array, intercept + slope * cos_sim_array, color='blue',
             linewidth=3)
    plt.xlabel('Cosine')
    plt.ylabel('Jaccard')
    plt.title('Cosine vs Jaccard')
    plt.savefig('results/jac_cos_regression.png')
    plt.close()
    logger.debug("End function")


def get_standard_deviations(term_freq_matrix, num_articles, num_features):
    """
    Gets the standard devaitions

    :param csr_matrix term_freq_matrix: the term frequence matrix
    :param int num_features: the number of features(terms) to use
    :param int num_articles: the number of articles
    """
    logger.debug("Begin function")
    # Standard deviation
    euc_sim_per_terms = {}
    cos_sim_per_terms = {}
    jac_sim_per_terms = {}
    euc_sd_per_terms = {}
    cos_sd_per_terms = {}
    jac_sd_per_terms = {}
    for num_feature in range(2, num_features):
        logger.debug('Progress: {}'.format(num_feature))

        (
            euc_sim_per_terms[num_feature], cos_sim_per_terms[num_feature],
            jac_sim_per_terms[num_feature]
        ) = get_similarities(term_freq_matrix, num_features)

        euc_sd_per_terms[num_feature] = numpy.std(
                [euc_sim_per_terms[num_feature][i][k] for
                 i, k in itertools.combinations(range(num_articles), 2)])
        cos_sd_per_terms[num_feature] = numpy.std([
            cos_sim_per_terms[num_feature][i][k] for
            i, k in itertools.combinations(range(num_articles), 2)])
        jac_sd_per_terms[num_feature] = numpy.std([
            jac_sim_per_terms[num_feature][i][k] for
            i, k in itertools.combinations(range(num_articles), 2)])

    plt.plot(euc_sd_per_terms.keys(), euc_sd_per_terms.values(), '-ro',
             label='Euclidean')
    plt.plot(cos_sd_per_terms.keys(), cos_sd_per_terms.values(), '-go',
             label='Cosine')
    plt.plot(jac_sd_per_terms.keys(), jac_sd_per_terms.values(), '-bo',
             label='Jaccard')
    plt.xlabel("Number of Features")
    plt.ylabel("Standard Deviation of Similarity Scores")
    plt.legend()
    plt.savefig('results/similarity_std.png')
    plt.close()
    logger.debug("End function")


def rank_articles(euc_list, cos_list, jac_list):
    logger.debug("Begin function")
    # Rank Articles
    euc_list.sort(key=lambda tup: tup[1], reverse=True)
    jac_list.sort(key=lambda tup: tup[1], reverse=True)
    cos_list.sort(key=lambda tup: tup[1], reverse=True)
    top_euc_pair = euc_list[:3]
    top_cos_pair = cos_list[:3]
    top_jac_pair = jac_list[:3]

    logger.info([[list(token_dict.keys())[art] for art in pair[0]]
                 for pair in top_euc_pair])
    logger.info([[list(token_dict.keys())[art] for art in pair[0]]
                 for pair in top_cos_pair])
    logger.info([[list(token_dict.keys())[art] for art in pair[0]]
                 for pair in top_jac_pair])
    logger.debug("End function")


def get_similarities(term_freq_matrix, num_features):
    """
    Returns the pairwise similarity matrices

    :param csr_matrix term_freq_matrix: the term frequence matrix
    :param int num_features: the number of features(terms) to use

    :return: the pairwise similarity matrices, in order of euclidean, cosine,
        jaccard
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    euc_sim = 1 / (1 + pairwise_distances(term_freq_matrix[:, :num_features],
                                          metric='euclidean', n_jobs=-1))
    cos_sim = 1 - pairwise_distances(term_freq_matrix[:, :num_features],
                                     metric='cosine', n_jobs=-1)
    jac_sim = pairwise_distances(term_freq_matrix[:, :num_features].toarray(),
                                 metric=gen_jaccard_similarity, n_jobs=-1)
    return euc_sim, cos_sim, jac_sim


def get_similarity_lists(euc_sim_matrix, cos_sim_matrix, jac_sim_matrix,
                         num_articles):
    """
    Returns the list of the similarity pairs in order of euclidean, cosine,
    jaccard

    :param numpy.ndarray euc_sim_matrix: pairwise similarity matrix
    :param numpy.ndarray cos_sim_matrix: pairwise similarity matrix
    :param numpy.ndarray jac_sim_matrix: pairwise similarity matrix
    :param int num_articles: the number of articles

    :return: returns the list of the similarity pairs in order of euclidean,
        cosine, jaccard
    :rtype: ([([int, int], float)], [([int, int], float)], [([int, int],
        float)])
    """
    euc_list = [([j, k], euc_sim_matrix[j][k]) for j, k in
                itertools.combinations(range(num_articles), 2)]
    cos_list = [([j, k], cos_sim_matrix[j][k]) for j, k in
                itertools.combinations(range(num_articles), 2)]
    jac_list = [([j, k], jac_sim_matrix[j][k]) for j, k in
                itertools.combinations(range(num_articles), 2)]

    return euc_list, cos_list, jac_list


def get_article_accuracy(y_true, y_pred, label):
    """
    Returns the accuracy of the specified label

    :param list y_true:
    :param list y_pred:
    :param str label:
    :return:
    """
    correct = 0
    total = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == label:
            if true_label == pred_label:
                correct += 1
            total += 1
    return correct / total


def setup_logger(logger_name):
    """
    Sets up the logger to be used for output messages
    :param str logger_name: the name to be used for the new logger
    :return: the new logger
    :rtype: logging.Logger
    """
    new_logger = logging.getLogger(logger_name)
    new_logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    full_handler = logging.StreamHandler(sys.stdout)
    full_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('term_frequency.log')
    file_handler.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    full_handler.setFormatter(
            logging.Formatter('{asctime} - {name} - {funcName} - '
                              '{levelname} - {message}',
                              datefmt='%m/%d/%y %I:%M:%S %p', style='{'))
    file_handler.setFormatter(
            logging.Formatter('{asctime} - {name} - {funcName} - '
                              '{levelname} - {message}',
                              datefmt='%m/%d/%y %I:%M:%S %p', style='{'))

    error_handler.setFormatter(
            logging.Formatter('{asctime} - {name} - {funcName} - '
                              '{lineno} - {levelname} - {message}',
                              datefmt='%m/%d/%y %I:%M:%S %p', style='{'))
    # add the handlers to the logger
    new_logger.addHandler(full_handler)
    new_logger.addHandler(file_handler)
    new_logger.addHandler(error_handler)
    return new_logger


logger = setup_logger('term_frequency')

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    logger.debug('Program started')
    stemmer = PorterStemmer()

    translator = str.maketrans('', '', string.punctuation)
    # Unwanted tags used for preprocessing with nltk to remove certain types
    # of words
    unwanted_tags = {'IN', 'PRP', 'PRP$', 'RB', 'RBS', 'RBR', 'RP', 'WADV',
                     'WD', 'WQ', 'WPRO', 'WPRO$'}

    contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
    path = 'testing'  # /alt.atheism'
    token_dict = {}

    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.path.sep + file
            article = open(file_path, 'r', errors='replace')
            article_text = article.read()
            article.close()
            article_text = article_text.lower()
            article_text = expand_contractions(article_text)
            article_text = article_text.translate(translator)
            token_dict[file_path] = article_text

    logger.debug('Finished reading files')

    tfidf_feat_sel = TfidfVectorizer(tokenizer=tokenize, stop_words='english',
                                     min_df=0.01, max_df=.99, max_features=100)
    tfs_feat_sel = csr_matrix(tfidf_feat_sel.fit_transform(token_dict.values()))
    labels_feat_sel = [re.split(r'/|\\', s)[-2]
                       for s in list(token_dict.keys())]

    tfidf_no_feat_sel = TfidfVectorizer(tokenizer=tokenize,
                                        stop_words='english')
    tfs_no_feat_sel = csr_matrix(
            tfidf_no_feat_sel.fit_transform(token_dict.values()))
    labels_no_feat_sel = [re.split(r'/|\\', s)[-2]
                          for s in list(token_dict.keys())]
    logger.debug('Finished creating term frequency')

    logger.debug("TFS Feature Selection Shape: Documents: {} Features: {}"
                 .format(*tfs_feat_sel.shape))
    logger.debug("TFS Without Feature Selection Shape: Documents: {} "
                 "Features: {}".format(*tfs_no_feat_sel.shape))

    counts = Counter()
    # Remove rows with mostly zeros
    to_remove = []
    for doc in range(tfs_feat_sel.shape[0]):
        row = tfs_feat_sel.getrow(doc)
        counts[row.nnz] += 1
        if row.nnz < 20:
            to_remove.append(doc)

    to_remove.reverse()
    for doc in to_remove:
        delete_row_csr(tfs_feat_sel, doc)
        del labels_feat_sel[doc]

    counts = Counter()
    # Remove rows with mostly zeros
    to_remove = []
    for doc in range(tfs_no_feat_sel.shape[0]):
        row = tfs_no_feat_sel.getrow(doc)
        counts[row.nnz] += 1
        if row.nnz < 20:
            to_remove.append(doc)

    to_remove.reverse()
    for doc in to_remove:
        delete_row_csr(tfs_no_feat_sel, doc)
        del labels_no_feat_sel[doc]

    logger.debug("TFS Feature Selection Shape: Documents: {} Features: {}"
                 .format(*tfs_feat_sel.shape))
    logger.debug("TFS Without Feature Selection Shape: Documents: {} "
                 "Features: {}".format(*tfs_no_feat_sel.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k_neighbors = KNeighborsClassifier(n_neighbors=4)
    decision_tree = DecisionTreeClassifier(random_state=42)

    scorers = {
        **{
            'article_accuracy_{}'.format(l):
                make_scorer(get_article_accuracy, label=l)
            for l in set(labels_feat_sel)
        },
        **{
            s: s for s in
            ['accuracy', 'f1_micro', 'precision_micro', 'recall_micro']
        }
    }
    logger.info("With feature selection")
    scores_tree = dict(cross_validate(decision_tree, tfs_feat_sel,
                                      labels_feat_sel, cv=cv, n_jobs=-1,
                                      return_train_score=False,
                                      scoring=scorers))

    scores_neighbors = dict(cross_validate(k_neighbors, tfs_feat_sel,
                                           labels_feat_sel, cv=cv, n_jobs=-1,
                                           return_train_score=False,
                                           scoring=scorers))
    logger.info('Decision Tree Classifier'
                + ''.join(['\n{}: {}'.format(k, v)
                           for k, v in scores_tree.items()]))
    logger.info('Nearest Neighbors Classifier'
                + ''.join(['\n{}: {}'.format(k, v)
                           for k, v in scores_neighbors.items()]))

    logger.info("Without feature selection")
    scores_tree2 = cross_validate(decision_tree, tfs_no_feat_sel,
                                  labels_no_feat_sel, cv=cv, n_jobs=-1,
                                  return_train_score=False,
                                  scoring=scorers)

    scores_neighbors2 = cross_validate(k_neighbors, tfs_no_feat_sel,
                                       labels_no_feat_sel, cv=cv, n_jobs=-1,
                                       return_train_score=False,
                                       scoring=scorers)

    logger.info('Decision Tree Classifier'
                + ''.join(['\n{}: {}'.format(k, v)
                           for k, v in scores_tree2.items()]))
    logger.info('Nearest Neighbors Classifier'
                + ''.join(['\n{}: {}'.format(k, v)
                           for k, v in scores_neighbors2.items()]))
