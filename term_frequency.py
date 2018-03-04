# !/usr/bin/env python3
"""
    Joshua Herman
    CS 6001
    Homework 3
"""

import itertools
import logging
import os
import pickle
import re
import string
import sys
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from scipy.sparse.csr import csr_matrix
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, make_scorer
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


def create_term_freq_histogram(term_freq_matrix):
    """
    Creates the term frequency histogram

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


def create_similarities_heatmap(euc_sim_matrix, cos_sim_matrix, jac_sim_matrix):
    """
    Creates the heatmap of similarities

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


def rank_articles(euc_list, cos_list, jac_list, token_dict):
    """
    Ranks the articles

    :param euc_list:
    :param cos_list:
    :param jac_list:
    :param token_dict:
    """
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


def get_similarities(term_freq_matrix, num_features=None):
    """
    Returns the pairwise similarity matrices

    :param csr_matrix term_freq_matrix: the term frequency matrix
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
    logger.debug('Begin Calculating Distances')
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
    logger.debug('Finish Calculating Distances')

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


def get_sse(term_freq_matrix, cluster_centers, cluster_assignments, dist_metric,
            num_clusters):
    """
    Calculates the sum of squares error

    :param csr_matrix term_freq_matrix: the term frequency matrix
    :param csr_matrix cluster_centers: the centers of the term
        frequency matrix clusters
    :param numpy.ndarray cluster_assignments: the cluster assignments of each
        article
    :param str dist_metric: the distance metric to use (`euclidean`,
        `cosine`, or `jaccard`)
    :param int num_clusters: the number of article clusters

    :return: the sum of squares error
    :rtype: float
    """
    sse = 0.0
    for i in range(num_clusters):
        cluster_i = term_freq_matrix[cluster_assignments == i, :]
        distance = get_center_distances(cluster_i, cluster_centers.getrow(i),
                                        dist_metric)
        sse += (distance ** 2).sum()
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
    winning_votes = [votes.most_common(1)[0] for votes in cluster_votes]

    pred_article_labels = numpy.array([winning_votes[cluster]
                                       for cluster in cluster_assignments])
    logger.debug("Finish Cluster Accuracy")
    return accuracy_score(article_labels, pred_article_labels)


def run_kmeans(term_freq_matrix, num_clusters, dist_metric,
               term_cond='centroids', num_iter=None):
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

    :return: the cluster assignments, number of iterations taken to complete
    :rtype: numpy.ndarray, int
    """
    # Randomly select initial clusters
    centroids = term_freq_matrix[
                (
                    numpy.random.choice(term_freq_matrix.shape[0], num_clusters,
                                        False)
                ), :]
    iteration = 0
    terminate = False
    assigned_clusters = None
    while not terminate:
        iteration += 1
        prev_centroid = centroids
        distances = get_center_distances(term_freq_matrix, centroids,
                                         dist_metric)
        assigned_clusters = numpy.array([dist.argmin() for dist in distances])
        centroids = get_cluster_centers(term_freq_matrix, assigned_clusters,
                                        num_clusters)
        if iteration % 10 == 0:
            logger.debug("Finished iteration {}".format(iteration))
        if (centroids != prev_centroid).nnz == 0:
            terminate = True
    return assigned_clusters, iteration


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

    :param list y_true: the true values
    :param list y_pred: the predicted values
    :param str label: the labels to get the accuracy of

    :return: the accuracy of the prediction
    :rtype: float
    """
    correct = 0
    total = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == label or label == 'all':
            if true_label == pred_label:
                correct += 1
            total += 1
    return correct / total


def compare_feature_selection(k_value, term_freq_matrix_with, labels_with,
                              term_freq_matrix_without, labels_without):
    """
    Compares the accuracy and f measures for KNN and Decision Tree
    Classifiers with and without feature selection

    :param int k_value: the k value to be used for KNN
    :param csr_matrix term_freq_matrix_with: the term freq matrix with
        feature selection used
    :param numpy.ndarray labels_with: the labels for term_term_freq_matrix_with
    :param csr_matrix term_freq_matrix_without: the term freq matrix without
        feature selection used
    :param numpy.ndarray labels_without: the labels for
        term_term_freq_matrix_without
    """
    logger.debug("Begin function")
    k_neighbors = KNeighborsClassifier(n_neighbors=k_value)
    decision_tree = DecisionTreeClassifier(random_state=42)
    scorers = ['accuracy', 'f1_weighted']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logger.info("With feature selection")
    scores_tree_with = dict(
            cross_validate(decision_tree, term_freq_matrix_with, labels_with,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers))

    scores_knn_with = dict(
            cross_validate(k_neighbors, term_freq_matrix_with, labels_with,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers)
    )

    avg_acc_tree_with = numpy.average(scores_tree_with['test_accuracy'])
    avg_f_tree_with = numpy.average(scores_tree_with['test_f1_weighted'])
    avg_acc_knn_with = numpy.average(scores_knn_with['test_accuracy'])
    avg_f_knn_with = numpy.average(scores_knn_with['test_f1_weighted'])

    scores_tree_without = dict(
            cross_validate(decision_tree, term_freq_matrix_without,
                           labels_without,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers))

    scores_knn_without = dict(
            cross_validate(k_neighbors, term_freq_matrix_without,
                           labels_without,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers)
    )

    avg_acc_tree_without = numpy.average(scores_tree_without['test_accuracy'])
    avg_f_tree_without = numpy.average(scores_tree_without['test_f1_weighted'])
    avg_acc_knn_without = numpy.average(scores_knn_without['test_accuracy'])
    avg_f_knn_without = numpy.average(scores_knn_without['test_f1_weighted'])

    logger.info('With Feature selection:\n'
                'Decision Tree Classifier: average accuracy: {}, '
                'average f-measure: {}'
                .format(avg_acc_tree_with, avg_f_tree_with))
    logger.info('With Feature selection:\n'
                'K Nearest Neighbors Classifier: average accuracy: {}, '
                'average f-measure: {}'
                .format(avg_acc_knn_with, avg_f_knn_with))

    logger.info('Without Feature selection:\n'
                'Decision Tree Classifier: average accuracy: {}, '
                'average f-measure: {}'
                .format(avg_acc_tree_without, avg_f_tree_without))
    logger.info('Without Feature selection:\n'
                'K Nearest Neighbors Classifier: average accuracy: {}, '
                'average f-measure: {}'
                .format(avg_acc_knn_without, avg_f_knn_without))
    logger.debug("End function")


def compare_classifiers(k_value, term_freq_matrix, labels_with):
    """
    Compares the accuracy and f measures for KNN and Decision Tree
    Classifiers and creates a histogram

    :param int k_value: the k value to be used for KNN
    :param csr_matrix term_freq_matrix: the term freq matrix with
        feature selection used
    :param numpy.ndarray labels_with: the labels for term_term_freq_matrix_with
    """
    logger.debug("Begin function")
    k_neighbors = KNeighborsClassifier(n_neighbors=k_value)
    decision_tree = DecisionTreeClassifier(random_state=42)
    scorers = ['accuracy', 'f1_weighted']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_tree = dict(
            cross_validate(decision_tree, term_freq_matrix, labels_with,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers))

    scores_knn = dict(
            cross_validate(k_neighbors, term_freq_matrix, labels_with,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers)
    )

    logger.info('Decision Tree Classifier:\n'
                'accuracies: {}\nf-measures: {}'
                .format(scores_tree['test_accuracy'],
                        scores_tree['test_f1_weighted']))
    logger.info('K Nearest Neighbors Classifier:\n'
                'accuracies: {}\nf-measures: {}'
                .format(scores_knn['test_accuracy'],
                        scores_knn['test_f1_weighted']))

    # Create histogram
    accuracies = ([('Tree{}'.format(k), scores_tree['test_accuracy'][k])
                   for k in range(5)]
                  + [('KNN{}'.format(k), scores_knn['test_accuracy'][k])
                     for k in range(5)])
    f_values = ([('Tree{}'.format(k), scores_tree['test_f1_weighted'][k])
                 for k in range(5)]
                + [('KNN{}'.format(k), scores_knn['test_f1_weighted'][k])
                   for k in range(5)])
    accuracies.sort(key=lambda x: x[1], reverse=True)
    f_values.sort(key=lambda x: x[1], reverse=True)

    x_pos = numpy.arange(len(accuracies))
    plt.bar(x_pos, [a[1] for a in accuracies], align='center', alpha=0.5)
    plt.xticks(x_pos, [a[0] for a in accuracies])
    plt.ylabel('Accuracy')
    plt.title('Classifiers Accuracy Comparison Histogram')
    plt.savefig('results/clf_comp_acc_histogram.png')
    plt.close()

    x_pos = numpy.arange(len(f_values))
    plt.bar(x_pos, [f[1] for f in f_values], align='center', alpha=0.5)
    plt.xticks(x_pos, [f[0] for f in f_values])
    plt.ylabel('F Measure')
    plt.title('Classifiers F-Measure Comparison Histogram')
    plt.savefig('results/clf_comp_fmeasure_histogram.png')
    plt.close()
    logger.debug("End function")


def compare_classifiers_articles(k_value, term_freq_matrix, labels):
    """
    Compares the accuracy and f measures for KNN and Decision Tree
    Classifiers and creates a histogram

    :param int k_value: the k value to be used for KNN
    :param csr_matrix term_freq_matrix: the term freq matrix with
        feature selection used
    :param numpy.ndarray labels: the labels for term_term_freq_matrix_with
    """
    logger.debug("Begin function")
    k_neighbors = KNeighborsClassifier(n_neighbors=k_value)
    decision_tree = DecisionTreeClassifier(random_state=42)
    scorers = {
        'article_accuracy_{}'.format(l):
            make_scorer(get_article_accuracy, label=l)
        for l in set(labels_feat_sel)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_tree = dict(
            cross_validate(decision_tree, term_freq_matrix, labels,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers))

    scores_knn = dict(
            cross_validate(k_neighbors, term_freq_matrix, labels,
                           cv=cv, n_jobs=-1, return_train_score=False,
                           scoring=scorers)
    )

    logger.info('Decision Tree Classifier accuracies:\n'
                + '\n'.join(['{}: {}'.format(article,
                                             scores_tree[
                                                 'test_article_'
                                                 'accuracy_{}'.format(article)])
                             for article in set(labels)]))
    logger.info('K Nearest Neighbors Classifier accuracies:\n'
                + '\n'.join(['{}: {}'.format(article,
                                             scores_knn[
                                                 'test_article_'
                                                 'accuracy_{}'.format(article)])
                             for article in set(labels)]))

    # Create histograms
    for article in set(labels):
        accuracies = (
                [('Tree{}'.format(k),
                  scores_tree['test_article_accuracy_{}'.format(article)][k])
                 for k in range(5)]
                + [('KNN{}'.format(k),
                    scores_knn['test_article_accuracy_{}'.format(article)][k])
                   for k in range(5)]
        )

        accuracies.sort(key=lambda x: x[1], reverse=True)

        x_pos = numpy.arange(len(accuracies))
        plt.bar(x_pos, [a[1] for a in accuracies], align='center', alpha=0.5)
        plt.xticks(x_pos, [a[0] for a in accuracies])
        plt.ylabel('Accuracy')
        plt.title('Classifiers Article {} Comparison Histogram'.format(article))
        plt.savefig('results/clf_acc_{}_histogram.png'.format(article))
        plt.close()

    logger.debug("End function")


def compare_k_value(term_freq_matrix, labels):
    """
    Evaluates how the K impacts the overall accuracy and f-measure of
    kNN on the dataset and plots histograms

    :param csr_matrix term_freq_matrix: the term frequence matrix
    :param numpy.ndarray labels: the list of labels pertaining to the
        term_freq_matrix

    :return: the best k value
    :rtype: int
    """
    logger.debug("Begin function")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorers = ['accuracy', 'f1_weighted']
    scores = []
    for k in range(1, 20):
        logger.debug("Testing k value: {}".format(k))
        k_neighbors = KNeighborsClassifier(n_neighbors=k, weights='distance')
        score = cross_validate(k_neighbors, term_freq_matrix, labels,
                               cv=cv, n_jobs=-1, return_train_score=False,
                               scoring=scorers)
        logger.info('Nearest Neighbors Classifier: k: {}'.format(k)
                    + ''.join(['\n{}: {}'.format(k, v)
                               for k, v in score.items()]))
        scores.append(score)

    avg_acc_k = [numpy.average(score['test_accuracy']) for score in scores]
    avg_f_k = [numpy.average(score['test_f1_weighted']) for score in scores]
    create_k_histogram(avg_acc_k, avg_f_k)
    accuracies = [(k + 1, avg_acc_k[k]) for k in range(len(avg_acc_k))]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    logger.debug("End function")
    return accuracies[0][0]


def create_k_histogram(avg_accuracy, avg_f_value):
    """
    Creates the histograms for comparing k values based on f measure and
    accuracy

    :param list avg_accuracy: the list of average accuracies as floats
    :param list avg_f_value: the list of average f measures as floats
    """
    logger.debug("Begin function")
    # Create histogram
    accuracies = [(k + 1, avg_accuracy[k]) for k in range(len(avg_accuracy))]
    f_values = [(k + 1, avg_f_value[k]) for k in range(len(avg_f_value))]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    f_values.sort(key=lambda x: x[1], reverse=True)

    x_pos = numpy.arange(len(accuracies))
    plt.bar(x_pos, [a[1] for a in accuracies], align='center', alpha=0.5)
    plt.xticks(x_pos, [a[0] for a in accuracies])
    plt.ylabel('Avg Accuracy')
    plt.xlabel('K Value')
    plt.title('K Value Accuracy Comparison Histogram')
    plt.savefig('results/k_comp_acc_histogram.png')
    plt.close()

    x_pos = numpy.arange(len(f_values))
    plt.bar(x_pos, [f[1] for f in f_values], align='center', alpha=0.5)
    plt.xticks(x_pos, [f[0] for f in f_values])
    plt.ylabel('Avg F Measure')
    plt.xlabel('K Value')
    plt.title('K Value F Measure Comparison Histogram')
    plt.savefig('results/k_comp_fmeasure_histogram.png')
    plt.close()
    logger.debug("End function")


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
        PATH = 'testing'  # '20news-18828'

        tfs_feat_sel, labels_feat_sel = get_term_frequency(PATH,
                                                           read_pickle=True,
                                                           write_pickle=False)
        (
            tfs_no_feat_sel, labels_no_feat_sel
        ) = get_term_frequency(PATH, feature_selection=False, read_pickle=True,
                               write_pickle=False)
        num_article_categories = len(set(labels_feat_sel))
        # Begin k means code
        resulting_clusters, iterations = run_kmeans(tfs_feat_sel,
                                                    num_article_categories,
                                                    'euclidean')
        logger.info('Number of iterations to complete: {}'.format(iterations))
    except Exception as exception:
        logger.exception(exception)
