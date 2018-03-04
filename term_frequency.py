# !/usr/bin/env python3
"""
    Joshua Herman
    CS 6001
    Homework 3
"""
import logging
import os
import pickle
import re
import string
import sys
from collections import Counter
from datetime import datetime

import nltk
import numpy
from nltk.stem.porter import PorterStemmer
from scipy.sparse.csr import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances

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
                    if token_tagged[1] not in UNWANTED_TAGS]
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
        logger.error(ValueError("works only for CSR format"
                                " use .tocsr() first"))
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


def get_center_distances(term_freq_matrix, cluster_centers, dist_metric):
    """
    Returns the distances from each row in the term_freq_matrix with
    each row in the term_freq_matrix_centers. Rows are article and columns
    are cluster numbers

    :param csr_matrix term_freq_matrix: the term frequency matrix
    :param csr_matrix cluster_centers: the centers of the term
        frequency matrix clusters
    :param str dist_metric: the distance metric to use (`euclidean`,
        `cosine`, or `jaccard`)

    :return: the pairwise distances matrices
    :rtype: numpy.ndarray
    """
    if dist_metric == 'euclidean':
        dist = pairwise_distances(term_freq_matrix, cluster_centers,
                                  metric='euclidean', n_jobs=-1)
    elif dist_metric == 'cosine':
        dist = pairwise_distances(term_freq_matrix, cluster_centers,
                                  metric='cosine', n_jobs=-1)
    elif dist_metric == 'jaccard':
        dist = 1 - pairwise_distances(term_freq_matrix.toarray(),
                                      cluster_centers.toarray(),
                                      metric=gen_jaccard_similarity, n_jobs=-1)
    else:
        raise ValueError("Invalid distance metric of {}. ".format(dist_metric)
                         + "Choices are: (euclidean, cosine, jaccard)")

    return dist


def get_cluster_centers(term_freq_matrix, cluster_assignments, num_clusters):
    """
    Calculates the new centers of each cluster

    :param csr_matrix term_freq_matrix: the term frequency matrix
    :param numpy.ndarray cluster_assignments: the cluster assignments of each
        article
    :param int num_clusters: the number of article clusters

    :return: the centers of each cluster
    :rtype: csr_matrix
    """
    centers = csr_matrix(
            numpy.vstack([
                term_freq_matrix[cluster_assignments == cluster, :].mean(0)
                for cluster in range(num_clusters)
            ])
    )
    return centers


def get_sse(cluster_assignments, distances):
    """
    Calculates the sum of squares error

    :param numpy.ndarray cluster_assignments: the cluster assignments of each
        article
    :param numpy.ndarray distances: the distances of each point from each
        centroid

    :return: the sum of squares error
    :rtype: float
    """
    sse = 0.0
    for i in range(len(cluster_assignments)):
        sse += distances[i][cluster_assignments[i]] ** 2

    return sse


def get_cluster_accuracy(cluster_assignments, article_labels, num_clusters):
    """
    Calculates the cluster accuracy

    :param numpy.ndarray cluster_assignments: the cluster assignments of each
        article
    :param numpy.ndarray article_labels: the true article labels
    :param int num_clusters: the number of article clusters

    :return: the accuracy of the clusters
    :rtype: float
    """
    logger.debug("Begin Cluster Accuracy")
    # Get the votes per cluster
    cluster_votes = [Counter() for _ in range(num_clusters)]
    for cluster, true_label in zip(cluster_assignments, article_labels):
        cluster_votes[cluster][true_label] += 1
    # Get the winning votes
    winning_votes = [votes.most_common(1)[0][0] for votes in cluster_votes]

    pred_article_labels = numpy.array([winning_votes[cluster]
                                       for cluster in cluster_assignments])

    logger.debug("Finish Cluster Accuracy")
    return accuracy_score(article_labels, pred_article_labels)


def run_kmeans(term_freq_matrix, num_clusters, dist_metric,
               term_cond='centroids', num_iter=100):
    """
    Performs k means clustering on the term frequency matrix

    :param csr_matrix term_freq_matrix: the term frequency matrix
    :param int num_clusters: the number of article clusters
    :param str dist_metric: the distance metric to use (`euclidean`,
        `cosine`, or `jaccard`)
    :param str term_cond: the termination condition (`centroids`, `sse`, `iter`)

        - `centroids`: terminate when there is no change in centroid position
        - `sse`: terminate when the SSE value increases in the next iteration
        - `iter`: terminate when the maximum preset value of iterations is
          complete
    :param int num_iter: the number of iterations to terminate at if
        ``term_cond`` is set to `iter`

    :return:  number of iterations taken to complete, the cluster
        assignments, centroids
    :rtype: int, numpy.ndarray, csr_matrix
    """
    logger.debug('Begin function')
    # Randomly select initial clusters
    centroids = term_freq_matrix[
                (
                    numpy.random.choice(term_freq_matrix.shape[0], num_clusters,
                                        False)
                ), :]

    iteration = 0
    assigned_clusters = None
    sse = None
    terminate_func = None
    if term_cond == 'centroids':
        def terminate_func():
            return (centroids != prev_centroid).nnz == 0
    elif term_cond == 'sse':
        def terminate_func():
            if prev_sse is not None and prev_sse == sse:
                # Terminate if clusters stayed the same since there won't be
                # any changes
                return (centroids != prev_centroid).nnz == 0
            return prev_sse is not None and prev_sse < sse
    elif term_cond == 'iter':
        def terminate_func():
            # Terminate if clusters stayed the same since there won't be
            # any changes
            if (centroids != prev_centroid).nnz == 0:
                return True
            return iteration >= num_iter

    while iteration == 0 or not terminate_func():
        iteration += 1
        distances = get_center_distances(term_freq_matrix, centroids,
                                         dist_metric)
        assigned_clusters = numpy.array([dist.argmin() for dist in distances])
        if term_cond == 'sse':
            prev_sse = sse
            sse = get_sse(assigned_clusters, distances)

        # elif term_cond == 'centroids':
        prev_centroid = centroids

        # Set the new centroids
        centroids = get_cluster_centers(term_freq_matrix, assigned_clusters,
                                        num_clusters)
        if iteration % 10 == 0:
            logger.debug("Finished iteration {}".format(iteration))
    logger.debug('End function')
    return iteration, assigned_clusters, centroids


def get_term_frequency(path, feature_selection=True, read_pickle=False,
                       write_pickle=False):
    """
    Returns the term frequency matrix and the associated article labels

    :param str path: the path to the articles
    :param bool feature_selection: whether or not to use feature selection
    :param bool read_pickle: whether or not to read from a pickle file
    :param bool write_pickle: whether or not to write to a pickle file

    :return: the term frequency matrix and the associated article labels
    :rtype: (csr_matrix, numpy.ndarray)
    """
    logger.debug("Begin function")
    pickle_file_name = 'pickles/{}/term_freq-with{}_feat_select.pkl'.format(
            path, ('' if feature_selection else 'out'))
    tfs = None
    labels = None
    if read_pickle:
        try:
            with open(pickle_file_name, 'rb') as pickle_file:
                tfs, labels = pickle.load(pickle_file)
            logger.debug("Successfully read {}".format(pickle_file_name))
        except FileNotFoundError as e:
            logger.warning(str(e) + '\n Recomputing instead')
            read_pickle = False

    if not read_pickle:
        token_dict = get_token_dict(path, read_pickle=True, write_pickle=False)

        tf_idf = TfidfVectorizer(tokenizer=tokenize, min_df=0.01, max_df=.99,
                                 stop_words='english')
        tfs = csr_matrix(tf_idf.fit_transform(token_dict.values()))
        labels = numpy.array([re.split(r'[/\\]', s)[-2]
                              for s in list(token_dict.keys())])

        logger.debug("TFS With{} Feature Selection Shape: Documents: {} "
                     "Features: {}"
                     .format(('' if feature_selection else 'out'), *tfs.shape))

        if feature_selection:
            clf = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                         n_jobs=-1, random_state=42)
            clf.fit(tfs, labels)
            feature_importance = numpy.array(clf.feature_importances_)
            col_to_remove = feature_importance.argsort()[:-100][::-1]
            tfs = csr_matrix(numpy.delete(tfs.toarray(), col_to_remove, 1))

            logger.debug("TFS With{} Feature Selection Shape: Documents: {} "
                         "Features: {}"
                         .format(('' if feature_selection else 'out'),
                                 *tfs.shape))

        counts = Counter()
        # Remove rows with mostly zeros
        to_remove = []
        for doc in range(tfs.shape[0]):
            row = tfs.getrow(doc)
            counts[row.nnz] += 1
            if row.nnz <= 10:
                to_remove.append(doc)

        to_remove.reverse()
        for doc in to_remove:
            delete_row_csr(tfs, doc)
            labels = numpy.delete(labels, doc)
        logger.debug('Finished creating term frequency')

    logger.debug("TFS With{} Feature Selection Shape: Documents: {} "
                 "Features: {}"
                 .format(('' if feature_selection else 'out'), *tfs.shape))
    if write_pickle:
        try:
            with open(pickle_file_name, 'wb') as pickle_file:
                pickle.dump((tfs, labels), pickle_file, pickle.HIGHEST_PROTOCOL)
            logger.debug("Successfully wrote {}".format(pickle_file_name))
        except FileNotFoundError as e:
            logger.warning('Unable to write\n' + str(e))
    logger.debug("End function")
    return tfs, labels


def get_token_dict(path, read_pickle=False, write_pickle=False):
    """
    Returns the dictionary of tokens

    :param string path: the path to the articles
    :param bool read_pickle: whether or not to read from a pickle file
    :param bool write_pickle: whether or not to write to a pickle file

    :return: the dictionary of tokens
    :rtype: dict
    """
    logger.debug("Begin function")
    pickle_file_name = 'pickles/{}/token_dict.pkl'.format(path)
    token_dictionary = None
    if read_pickle:
        try:
            with open(pickle_file_name, 'rb') as pickle_file:
                token_dictionary = pickle.load(pickle_file)
            logger.debug("Successfully read {}".format(pickle_file_name))
        except FileNotFoundError as e:
            logger.warning(str(e) + '\n Recomputing instead')
            read_pickle = False
    if not read_pickle:
        token_dictionary = {}
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = '{}/{}'.format(subdir, file)
                article = open(file_path, 'r', errors='replace')
                article_text = article.read()
                article.close()
                article_text = article_text.lower()
                article_text = expand_contractions(article_text)
                article_text = article_text.translate(translator)
                token_dictionary[file_path] = article_text
    if write_pickle:
        try:
            with open(pickle_file_name, 'wb') as pickle_file:
                pickle.dump(token_dictionary, pickle_file,
                            pickle.HIGHEST_PROTOCOL)
                logger.debug("Successfully wrote {}".format(pickle_file_name))
        except FileNotFoundError as e:
            logger.warning('Unable to write\n' + str(e))
    logger.debug("End function")
    return token_dictionary


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
    try:
        numpy.random.seed(42)
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        logger.debug('Program started')

        stemmer = PorterStemmer()

        translator = str.maketrans('', '', string.punctuation)
        # Unwanted tags used for preprocessing with nltk to remove certain types
        # of words
        UNWANTED_TAGS = {'IN', 'PRP', 'PRP$', 'RB', 'RBS', 'RBR', 'RP', 'WADV',
                         'WD', 'WQ', 'WPRO', 'WPRO$'}

        contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
        PATH = '20news-18828'

        tfs_feat_sel, labels_feat_sel = get_term_frequency(PATH,
                                                           read_pickle=True,
                                                           write_pickle=False)
        (
            tfs_no_feat_sel, labels_no_feat_sel
        ) = get_term_frequency(PATH, feature_selection=False, read_pickle=True,
                               write_pickle=False)
        k = len(set(labels_feat_sel))
        # Begin k means code
        resulting_clusters = dict()
        iterations = dict()
        for term_condition in ['centroids', 'sse', 'iter']:
            for metric in ['euclidean', 'cosine', 'jaccard']:
                start_time = datetime.now()
                (
                    iterations, result_clusters, result_centroids
                ) = run_kmeans(tfs_feat_sel, k, metric, term_condition)
                run_time = datetime.now() - start_time

                resulting_sse = get_sse(result_clusters,
                                        get_center_distances(tfs_feat_sel,
                                                             result_centroids,
                                                             metric))
                resulting_accuracies = get_cluster_accuracy(result_clusters,
                                                            labels_feat_sel, k)
                logger.info('With Feature Selection\n'
                            'Distance metric: {}; Termination Condition: {}; '
                            'Iterations: {}; SSE: {}; Acc: {}; Run Time: {}'
                            .format(metric, term_condition, iterations,
                                    resulting_sse, resulting_accuracies,
                                    run_time))

        for term_condition in ['centroids']:
            for metric in ['euclidean', 'cosine', 'jaccard']:
                start_time = datetime.now()
                (
                    iterations, result_clusters, result_centroids
                ) = run_kmeans(tfs_no_feat_sel, k, metric, term_condition)
                run_time = datetime.now() - start_time
                resulting_accuracies = get_cluster_accuracy(result_clusters,
                                                            labels_no_feat_sel,
                                                            k)
                logger.info('Without Feature Selection\n'
                            'Distance metric: {}; Termination Condition: {}; '
                            'Iterations: {}; Acc: {}; Run Time: {}'
                            .format(metric, term_condition, iterations,
                                    resulting_accuracies,
                                    run_time))

    except Exception as exception:
        logger.exception(exception)
