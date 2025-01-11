'''
kmeans clustering using modules from scikit learn. Uses the inverted index 
to compile tf-idf scores which is then used to compute the clusters using the 
specified k. Silhouette score is used to assess the clustering performance.
'''
import os
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cwd = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.realpath(cwd))

INDEX_DIRECTORY = "/".join([root, 'INDEX'])
POSITIONAL_INDEX = "index"
INDEX_SUFFIX = ".txt"
POSITIONAL_INDEX = "/".join([INDEX_DIRECTORY,
                             POSITIONAL_INDEX + INDEX_SUFFIX])

def read_positional_index(output_file):
    """Reads the positional index from the file and returns it as a dictionary."""
    positional_index = {}
    with open(POSITIONAL_INDEX, "r", encoding="utf-8", errors='ignore') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or " " not in line:
                # Empty/missing lines
                continue
            try:
                term, postings_str = line.split(maxsplit=1)
                matches = re.findall(r'(\d+)\[([\d,]+)\]', postings_str)
                positional_index[term] = {int(doc_id): list(
                    map(int, pos.split(','))) for doc_id, pos in matches}
            except ValueError as e:
                output_file.write(
                    f"Skipping malformed line: {line} (Error: {e})\n")
    return positional_index

# Step 1: Convert the positional index to a term-document matrix
def build_term_document_matrix(positional_index):
    """Builds a term-document matrix from the positional index."""
    terms = list(positional_index.keys())
    doc_ids = set(doc_id for postings in positional_index.values()
                  for doc_id in postings.keys())
    doc_ids = sorted(doc_ids)  # Ensure consistent ordering

    term_index = {term: i for i, term in enumerate(terms)}
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Create a term-document matrix
    tdm = np.zeros((len(terms), len(doc_ids)), dtype=np.float32)
    for term, postings in positional_index.items():
        term_idx = term_index[term]
        for doc_id, positions in postings.items():
            doc_idx = doc_index[doc_id]
            tdm[term_idx, doc_idx] = len(positions)  # Use term frequency

    return tdm, terms, doc_ids

# Step 2: Apply TF-IDF normalization
def apply_tfidf_normalization(tdm):
    """Applies TF-IDF normalization to the term-document matrix."""
    tfidf_transformer = TfidfTransformer(norm='l2')
    tdm_tfidf = tfidf_transformer.fit_transform(
        tdm.T)  # Transpose to get documents as rows
    return tdm_tfidf

# Step 3: Cluster the documents using KMeans
def cluster_documents(tdm_tfidf, k):
    """Clusters the documents using KMeans."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(tdm_tfidf)
    return kmeans

# Step 4: Evaluate clustering performance
def evaluate_clustering(tdm_tfidf, kmeans):
    """Evaluates clustering performance using silhouette score."""
    labels = kmeans.labels_
    score = silhouette_score(tdm_tfidf, labels)
    return score

# Step 5: Extract top terms for each cluster
def get_top_terms_by_cluster(tdm_tfidf, terms, kmeans, top_n=20):
    """Get the top vocabulary terms for each cluster ranked by TF-IDF."""
    clusters = kmeans.labels_
    num_clusters = kmeans.n_clusters
    cluster_terms = {}

    # Convert sparse TF-IDF matrix to dense for easy manipulation
    tdm_tfidf_dense = tdm_tfidf.todense()

    for cluster in range(num_clusters):
        # Get document indices for the current cluster
        doc_indices = [i for i, label in enumerate(
            clusters) if label == cluster]

        # Aggregate TF-IDF scores for terms in the cluster
        cluster_tfidf_sum = tdm_tfidf_dense[doc_indices, :].sum(axis=0)

        # Flatten the array and sort terms by their scores
        cluster_tfidf_scores = np.array(cluster_tfidf_sum).flatten()
        top_term_indices = cluster_tfidf_scores.argsort(
        )[-top_n:][::-1]  # Indices of top terms

        # Map indices back to terms
        cluster_terms[cluster] = [
            (terms[idx], cluster_tfidf_scores[idx]) for idx in top_term_indices]

    return cluster_terms

def display_top_terms(cluster_terms, output_file):
    """Display the top terms for each cluster."""
    for cluster, terms in cluster_terms.items():
        output_file.write(f"\nCluster {cluster}:\n")
        for term, score in terms:
            output_file.write(f"{term}: {score:.4f}\n")


def summarize_clusters(doc_ids, labels, output_file):
    """Summarizes the number of documents in each cluster."""
    cluster_counts = Counter(labels)
    summary = {cluster: count for cluster, count in cluster_counts.items()}
    output_file.write("\nCluster Summary:\n")
    for cluster, count in summary.items():
        output_file.write(f"Cluster {cluster}: {count} documents\n")
    return summary

def main():
    '''
    Main function
    '''
    output_clustering_results = "/".join([INDEX_DIRECTORY,
                                          'clustering_results' + INDEX_SUFFIX])
    with open(output_clustering_results, "w", encoding='utf-8') as output_file:
        positional_index = read_positional_index(output_file)
        tdm, terms, doc_ids = build_term_document_matrix(positional_index)

        # Apply TF-IDF normalization
        tdm_tfidf = apply_tfidf_normalization(tdm)

        # Number of departments and faculties/schools
        num_departments = 110  # Actual number of departments = 81
        num_faculties = 50  # Actual number of schools = 10

        # Clustering for departments
        kmeans_departments = cluster_documents(tdm_tfidf, num_departments)
        score_departments = evaluate_clustering(tdm_tfidf, kmeans_departments)
        output_file.write(
            f"Silhouette Score for Departments Clustering (k={num_departments}): {score_departments}\n")

        # Summarize clusters for departments
        output_file.write("\nClustering Results for Departments:\n")
        summarize_clusters(doc_ids, kmeans_departments.labels_, output_file)

        # Get top terms for department clusters
        top_terms_departments = get_top_terms_by_cluster(
            tdm_tfidf, terms, kmeans_departments)
        output_file.write("\nTop 50 Terms for Each Department Cluster:\n")
        display_top_terms(top_terms_departments, output_file)

        # Clustering for faculties
        kmeans_faculties = cluster_documents(tdm_tfidf, num_faculties)
        score_faculties = evaluate_clustering(tdm_tfidf, kmeans_faculties)
        output_file.write(
            f"Silhouette Score for School Clustering (k={num_faculties}): {score_faculties}\n")

        # Summarize clusters for schools
        output_file.write("\nClustering Results for Schools:\n")
        summarize_clusters(doc_ids, kmeans_faculties.labels_, output_file)

        # Get top terms for school clusters
        top_terms_faculties = get_top_terms_by_cluster(
            tdm_tfidf, terms, kmeans_faculties)
        output_file.write("\nTop 50 Terms for Each School Cluster:\n")
        display_top_terms(top_terms_faculties, output_file)


if __name__ == "__main__":
    main()
