import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, EditedNearestNeighbours
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def load():
    train_transaction = pd.read_csv('../train_transaction.csv', index_col='TransactionID')
    test_transaction = pd.read_csv('../test_transaction.csv', index_col='TransactionID')

    train_identity = pd.read_csv('../train_identity.csv', index_col='TransactionID')
    test_identity = pd.read_csv('../test_identity.csv', index_col='TransactionID')

    sample_submission = pd.read_csv('../sample_submission.csv', index_col='TransactionID')

    train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

    print(train.shape)
    print(test.shape)

    y_train = train['isFraud'].copy()
    del train_transaction, train_identity, test_transaction, test_identity

    # Drop target, fill in NaNs
    X_train = train.drop('isFraud', axis=1)
    X_test = test.copy()

    del train, test

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # Label Encoding
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values))
    return X_train, y_train, X_test, sample_submission

def sampling(X_train, y_train):
    ran_over = RandomOverSampler(random_state=42)
    X_train_oversample,y_train_oversample = ran_over.fit_resample(X_train,y_train)
    ran_under = RandomUnderSampler(random_state=42)
    X_train_undersample, y_train_undersample = ran_under.fit_resample(X_train,y_train)
    tl = TomekLinks(n_jobs=6)
    X_train_tl, y_train_tl = tl.fit_sample(X_train, y_train)
    sm = SMOTE(random_state=42, n_jobs=5)
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
    enn = EditedNearestNeighbours()
    X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

    print(np.unique(y_train, return_counts=True))
    print("after sampling")
    print("randomg over sampling")
    print(np.unique(y_train_oversample, return_counts=True))
    print("SMOTE sampling")
    print(np.unique(y_train_sm, return_counts=True))
    print("random under sampling")
    print(np.unique(y_train_undersample, return_counts=True))
    print("TomekLinks under sampling")
    print(np.unique(y_train_tl, return_counts=True))
    return (X_train_oversample, y_train_oversample, X_train_undersample, y_train_undersample,
     X_train_tl, y_train_tl, X_train_sm, y_train_sm, X_train_enn, y_train_enn)


def plot_2d_space(X_train, y_train,X,y ,label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(8,4))
   
    for l, c, m in zip(np.unique(y), colors, markers):
        ax1.scatter(
            X_train[y_train==l, 0],
            X_train[y_train==l, 1],
            c=c, label=l, marker=m
        )
    for l, c, m in zip(np.unique(y), colors, markers):
        ax2.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
   
    ax1.set_title(label)
    ax2.set_title('original data')
    plt.legend(loc='upper right')
    plt.show()

# visualization
# T-SNE Implementation
def dimension_reduction(X_train):
    t0 = time.time()
    X_reduced_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(X_train)
    t1 = time.time()
    print("T-SNE took {:.2} s".format(t1 - t0))

    # PCA Implementation
    t0 = time.time()
    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_train)
    t1 = time.time()
    print("PCA took {:.2} s".format(t1 - t0))

    # TruncatedSVD
    t0 = time.time()
    X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X_train)
    t1 = time.time()
    print("Truncated SVD took {:.2} s".format(t1 - t0))
    return (X_reduced_tsne, X_reduced_pca, X_reduced_svd)

def scatter_plot(X_train, y_train):
	X_reduced_tsne, X_reduced_pca, X_reduced_svd = dimension_reduction(X_train)
	random_index = np.random.randint(1,len(y_train),1000) # random sample 1000 data points for visualization
	y = np.array(y_train)[random_index]
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
	# labels = ['No Fraud', 'Fraud']
	f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

	blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
	red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

	# t-SNE scatter plot
	ax1.scatter(X_reduced_tsne[random_index,0], X_reduced_tsne[random_index,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
	ax1.scatter(X_reduced_tsne[random_index,0], X_reduced_tsne[random_index,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
	ax1.set_title('t-SNE', fontsize=14)

	ax1.grid(True)

	ax1.legend(handles=[blue_patch, red_patch])

	# PCA scatter plot
	ax2.scatter(X_reduced_pca[random_index,0], X_reduced_pca[random_index,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
	ax2.scatter(X_reduced_pca[random_index,0], X_reduced_pca[random_index,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
	ax2.set_title('PCA', fontsize=14)

	ax2.grid(True)

	ax2.legend(handles=[blue_patch, red_patch])

	# TruncatedSVD scatter plot
	ax3.scatter(X_reduced_svd[random_index,0], X_reduced_svd[random_index,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
	ax3.scatter(X_reduced_svd[random_index,0], X_reduced_svd[random_index,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
	ax3.set_title('Truncated SVD', fontsize=14)

	ax3.grid(True)

	ax3.legend(handles=[blue_patch, red_patch])

	plt.show()

def main(vis=False, samp=False):
	print("load data")
	X_train, y_train, X_test, sample_submission = load()

	if samp:
		print("sampling")
		X_train_oversample, y_train_oversample, X_train_undersample, y_train_undersample, X_train_tl, y_train_tl, X_train_sm, y_train_sm, X_train_enn, y_train_enn = sampling(X_train, y_train)
	
	print("training")
	# classification
	clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    missing=-999,
    random_state=2020,
    n_jobs=5,
    tree_method='hist', 
    # scale_pos_weight=27
    )
	clf.fit(X_train, y_train) # X_train_oversample, y_train_oversample, X_train_undersample, y_train_undersample, X_train_tl, y_train_tl, X_train_sm, y_train_sm, X_train_enn, y_train_enn
	sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
	sample_submission.to_csv('../simple_xgboost.csv')
	if vis:
		print("visualization")
		scatter_plot(X_train, y_train) # X_train_tl, X_train_sm, X_train_enn # y_train_tl, y_train_sm, y_train_enn

if __name__ == '__main__':
	main()