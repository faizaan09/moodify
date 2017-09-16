#!/usr/bin/python

import sqlite3
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.corpus import stopwords
from sklearn.decomposition import RandomizedPCA




def dict_factory(cursor, row):
	d = {}
	for idx, col in enumerate(cursor.description):
		d[col[0]] = row[idx]
	return d







#print "Opened database successfully"
def getData():
	conn = sqlite3.connect('growing_training.sqlite')
	conn.row_factory = dict_factory
	cursor = conn.cursor()
	cursor.execute("SELECT * FROM moodtable;")
	result = cursor.fetchall()

	lyrics_data = []
	mood_data = []

	for data in result:
		lyrics_data.append(data['lyrics'].encode('ascii','ignore'))
		if data['mood'].encode('ascii','ignore') == "happy":
			mood_data.append(1)
		else:
			mood_data.append(0)

	stemmer = SnowballStemmer("english")
	##print lyrics_data[0]
	#print "ffffffffddddddddd"






	for i in range(len(lyrics_data)):
		text = stopwords.words("english") 
	        for p in text:
	        	lyrics_data[i] = lyrics_data[i].replace(p,"")


	
	for i in range(len(lyrics_data)):
		stemmed = [stemmer.stem(w) for w in lyrics_data[i].split()]
		lyrics_data[i] = " ".join(stemmed)




	### test_size is the percentage of events assigned to the test set
	### (remainder go into training)
	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(lyrics_data, mood_data, test_size=0.1, random_state=42)


	pca = RandomizedPCA().fit(features_train)

	# eigenfaces = pca.components_.reshape((n_components, h, w))

	# print "Projecting the input data on the eigenfaces orthonormal basis"
	features_train = pca.transform(features_train)
	features_test = pca.transform(features_test)

	### text vectorization--go from strings to lists of numbers
	vectorizer = TfidfVectorizer(sublinear_tf=True,
								 stop_words='english')
	features_train_transformed = vectorizer.fit_transform(features_train)
	features_test_transformed  = vectorizer.transform(features_test)



	### feature selection, because text is super high dimensional and 
	### can be really computationally chewy as a result
	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(features_train_transformed, labels_train)
	features_train_transformed = selector.transform(features_train_transformed).toarray()
	features_test_transformed  = selector.transform(features_test_transformed).toarray()

	## info on the data
	print "no. of happy songs:", sum(labels_train)
	print "no. of sad songs:", len(labels_train)-sum(labels_train)

	conn.close()
	
	return features_train_transformed, features_test_transformed, labels_train, labels_test


# getData()







