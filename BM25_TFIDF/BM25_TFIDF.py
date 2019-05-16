from __future__ import unicode_literals
import os
import tqdm
import struct
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
import unicodedata
import json


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Implement encoding and decoding function for the lossy compression, as done here 
# https://github.com/apache/lucene-solr/blob/master/lucene/core/src/java/org/apache/lucene/util/SmallFloat.java
def floatToRawIntBits(f):
    s = struct.pack('=f', f)
    return struct.unpack('=l', s)[0]

def intBitsToFloat(b):
    s = struct.pack('>l', b)
    return struct.unpack('>f', s)[0]

def byte315ToFloat(b):
    if (b == 0):
        return 0.0
    bits = (b&0xff) << (24-3)
    bits += (63-15) << 24
    return intBitsToFloat(bits)

def floatToByte315(f):
    bits = floatToRawIntBits(f)
    
    smallfloat = bits >> (24-3)
    
    if (smallfloat <= ((63-15)<<3)):
        return  bytes(0) if (bits<=0) else bytes(1)
    
    if (smallfloat >= ((63-15)<<3) + 0x100):
        return -1
    
    return int(bytes(smallfloat - ((63-15)<<3)))

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class Similarity(object):
    def __init__(self): 
        self.NORM_TABLE = np.arange(256, dtype= float)
    
    def decodeNormValue(self, b):
        return self.NORM_TABLE[b & 0xFF] # & 0xFF maps negative bytes to positive above 127
    
    def encodeNormValue(self, value):
        pass
    
    def get_idf(self, docCount, docFreqs):
        pass
    
    def get_tf(self, docs, query):
    	# with open('/home/jiangkun/MasterThesis/Notes/BM25_TFIDF/term2docFreqs.json') as f:
    		# data = json.load(f)
        vect = CountVectorizer(vocabulary = query, analyzer='word') # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        tf = vect.fit_transform(docs) # tf.shape >> (8 ,2) document-term matrix which is sparse matrix, value os term occurences
        # print(tf.toarray())
        names = vect.get_feature_names()
        # each_name_vector = []
        # for i in names:
        	# each_name_vector.append(data[i])
        # print(names)
        # tf = sparse.csr_matrix(each_name_vector).T ## np.matrix has no attribute toarray()
        # print(tf.toarray())
        # print(tf[0,:])
        # print(tf.shape)
        docCount = tf.shape[0] #### number of docs
        docFreqs = (tf != 0).sum(0) #### number of docs containing this term
        # print(type(docFreqs))
        # print(docFreqs)
        return (tf, docCount, docFreqs)



        ################
        # each_term_vector = []
        # for i in query:
        # 	each_term_vector.append(self.doc_mat.toarray()[:, i])

        # tf = sp.csr_matrix(each_term_vector).T
        # docCount = tf.shape[0]
        # return (tf, docCount)
    
    def execute(self, query, docs):
        pass
    
    def transform(self, value):
        pass
    
    def get_coord(self, tf, idf):
        nTerms = tf.shape[1]
        return np.divide((tf != 0).astype(float)*(idf != 0).astype(float), nTerms)
    
    def get_norm(self, docs):
        vect = CountVectorizer(analyzer='word')
        X = vect.fit_transform(docs) #### the terms come from all the unique term from all docs. 
                                     # X.shape >> (8, 14) the sequence of terms is based on alphabet
        # print(X.toarray())
        transpose_X = X.toarray().T
        # print(transpose_X)

        # print(X.shape)
        # docFreqs = (X != 0 ).sum(0).flat
        # print(docFreqs)
        names = vect.get_feature_names()
        # print(names)
        # term2vector = {}
        # for i, j in enumerate(range(transpose_X.shape[0])):
        	# term2vector[names[i]] = transpose_X[i, :].tolist()
        # print(term2vector)
        # with open('/home/jiangkun/MasterThesis/Notes/BM25_TFIDF/term2docFreqs.json', 'w') as f:
        	# f.write(json.dumps(term2vector))
        # print(names)
        docCount = X.shape[0]

        avgFieldLength = X.sum()/float(docCount)
        each_doc_length = (X != 0).sum(1) # np.matrix
        # print(type(avgFieldLength))

        norm = each_doc_length
        
        # norm = np.matrix(map(self.decodeNormValue, 
                             # map(self.encodeNormValue, 
                                 # map(self.transform, X.sum(axis = 1))))).reshape(docCount, 1)
        # print(norm)
        
        return (norm, avgFieldLength)


        
    
    def score(self, query, docs):
    	query = normalize(query)
    	query = query.split(' ')
    	scores = self.execute(query, docs)
    	return scores


class ClassicSimilarity(Similarity):
    def __init__(self):
        Similarity.__init__(self)
    
        # for i in range(256):
        #     self.NORM_TABLE[i] = byte315ToFloat(int(bytes(i)))

    def encodeNormValue(self, value):
        return floatToByte315(value)
    
    def get_idf(self, docCount, docFreqs):
        idf = 1.0 + np.log(np.divide(docCount, (docFreqs + 1.0)))
        # print('original idf: ', idf)
        return np.square(idf, idf).T
    
    def get_coord(self, tf, idf):
        nTerms = tf.shape[1]
        return np.divide((tf != 0).astype(float)*(idf != 0).astype(float), nTerms)
    
    def transform(self, value):
        return 1.0/np.sqrt(value)
        
    def execute(self, query, docs):
        tf, docCount, docFreqs = self.get_tf(docs, query)
        idf = self.get_idf(docCount, docFreqs)
        # print('idf: ', idf)
        tf = tf.sqrt()
        coord = self.get_coord(tf, idf)
        norm, avgFieldLength = self.get_norm(docs)
        print(norm)
        queryNorm = np.divide(1.0, np.sqrt(idf.sum(axis = 0)))
        # coord * dot(tf * norm, idf * queryNorm)
        # tf = tf.multiply(norm)
        idf = np.multiply(idf, queryNorm)
        tfidf = tf.dot(idf)
        tfidf = np.multiply(tfidf, coord)
        print(tfidf)
        tfidf = np.multiply(tfidf, np.divide(1.0, np.sqrt(norm))) ##########
        # print(type(tfidf))
        return tfidf ## np.matrix


class BM25(Similarity):
    def __init__(self, k = 1.2, b = 0.75, coord_factor = True): ## coordq, d : is a score factor based on how many of the query terms are found in the specified document. 
                                                                # For example, if documnt d contains 2 of the 3 terms of query q, then coordq, d is 2/3.
        Similarity.__init__(self)
        self.k = k
        self.b = b
        self.coord_factor = coord_factor # multiply the scores by the coord factor: n query terms in the document / total n of terms in the query
    
        # for i in range(1, 256):
        #     f = byte315ToFloat(int(bytes(i)))
        #     self.NORM_TABLE[i] = 1.0 / (f*f)
        # self.NORM_TABLE[0] = 1.0 / self.NORM_TABLE[255]

    def encodeNormValue(self, value):
        boost = 1.0
        return floatToByte315(boost / float(np.sqrt(value)))
    
    def get_idf(self, docCount, docFreqs):
        idf = np.log(1 + np.divide((docCount - docFreqs + 0.5), (docFreqs + 0.5)))
        return idf.T
    
    def transform(self, value):
        return value
        
    def execute(self, query, docs):
        tf, docCount, docFreqs = self.get_tf(docs, query)
        fieldLengths, avgFieldLength = self.get_norm(docs)
        # print(fieldLengths)
        # print(avgFieldLength)
        tfNorm = np.divide((tf * (self.k + 1)).toarray(),
                           (tf + self.k * (1 - self.b + self.b * (fieldLengths/avgFieldLength))))
        idf = self.get_idf(docCount, docFreqs)
        if self.coord_factor == True:
            coord = self.get_coord(tf, idf)
        else:
            coord = 1.0
        # dot(tf, idf)
        tfidf = tfNorm.dot(idf)
        # print('xixi: ', type(np.multiply(tfidf, coord)))
        return np.multiply(tfidf, coord) ### np.matrix


############################
        # tf, docCount, docFreqs = self.get_tf(query)


def print_rank(docs, scores):
    candidates = scores != 0
    # print(candidates)
    # print(np.asarray(candidates.T)[0,:])
    indices = docs.index[np.asarray(candidates.T)[0,:]]
    # print(indices)
    sorted_indices = indices[np.argsort(-np.asarray(scores[indices].T)[0,:])]
    # print(-np.asarray(scores[indices].T)[0,:])
    # print(np.argsort(-np.asarray(scores[indices].T)[0,:])) ### minus ==> default situation is ascending, so that the largest will become the smallest and at the first position
    # print(sorted_indices)
    for rank, idx in enumerate(sorted_indices):
        print("%d.\t%s\t%f" %(rank, docs[idx], scores[idx, 0]))


if __name__ == '__main__':

	# Some dummies documents
	docs = pd.Series(["Lucene\n\nAction\n\ncontinue continued", "Lucene\n\nDummies mbuy haha haha haha haha", "Managing Gigabytes", "Art  Computer Science", "Action", "Lucene way", "Managing Megabytes lucene", "Art Gaming", "Brazil\n\nBrazil (; ), officially the"])
	doc_ids = pd.Series(['hehe', 'dada', 'xixi', 'meme', 'zz', 'xx', 'cc', 'vv'])
	# bm25 = BM25()
	# scores = bm25.score('lucene action', docs)
	# print(scores)
	# print(len(scores.data))
	# print_rank(docs, scores)

	tf_idf = ClassicSimilarity()
	scores_tfidf = tf_idf.score('lucene action', docs)
	print_rank(docs, scores_tfidf)