from scipy.sparse import csr_matrix
import numpy as np

class TfidfVectorizerScratch:
  def __init__(self):
    pass
  def _calculate_tf_per_doc(self,doc):
    tokens = doc.split()
    unique_tokens = set(tokens)
    tf_dict = {
        token: tokens.count(token)/len(tokens)
        for token in unique_tokens
    }
    return tf_dict
  
  def _calculate_df(self,many_doc):
    # many_doc is a list
    many_doc_splitted = [document.split() for document in many_doc]
    # concat all the document
    concatenated_doc = " ".join(many_doc)
    # create the set of unique tokens
    unique_tokens = list(set(concatenated_doc.split()))
    def _count_df_per_doc(token,many_doc_splitted):
      res = 0
      for doc in many_doc_splitted:
        if token in doc:
          res += 1
      return res
    # create the df dict
    df_dict = {
        token: _count_df_per_doc(token,many_doc_splitted)
        for token in unique_tokens
    }
    return df_dict
  
  def _calculate_idf(self,N,df_dict):
    return {
        token: np.log10(N/df_i) for token, df_i in df_dict.items()
    }
  
  def _calculate_tf_idf_per_doc(self,tf_per_doc,idf):
    res = {}
    for token in tf_per_doc.keys():
      idf_token = idf[token] if token in idf.keys() else 0
      res[token] = tf_per_doc[token] * idf_token
    return res
  
  def _transform_tfidfdict_to_tfidfarray(self,tfidf_dict):
    # processing one row / one doc
    # initialize with 0
    res = [0 for k in self.idfs.keys()]
    # fill the 0s
    for k in tfidf_dict.keys():
      if k in self.idfs.keys():
        try:
          idx = self.vocabulary_[k]
          res[idx] = tfidf_dict[k]
        except:
          pass
    return res
  
  def fit_transform(self,docs):
    # fit
    self.fit(docs)
    # transform
    tfidfs = self.transform(docs)
    return tfidfs
  
  def fit(self,docs):
    # calculate df
    df_dict = self._calculate_df(docs)
    # calculate idf
    self.idfs = self._calculate_idf(len(docs),df_dict)
    # save vocabulary
    self.vocabulary_ = {
        token: i for i,token in enumerate(self.idfs.keys())
    }
    return self
  
  def transform(self,docs):
    if not self.idfs: # not fitted yet
      raise Exception("Vectorizer is not fitted yet")
    # else
    # calculate tf
    tfs = [
           self._calculate_tf_per_doc(doc) for doc in docs
    ]
    # calculate tfidf
    tfidfs = [
              self._calculate_tf_idf_per_doc(tf,self.idfs) for tf in tfs
    ]
    # transform to array
    tfidfs = [
              self._transform_tfidfdict_to_tfidfarray(el) for el in tfidfs
    ]
    # convert to sparse matrix
    tfidfs = csr_matrix(tfidfs)
    return tfidfs