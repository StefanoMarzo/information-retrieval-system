
#imports
import xml.etree.ElementTree as ET
import os
import string
import numpy as np
import spacy #lemmatization
import pickle #serialize object read/write files
import time


#Load document collection

#path to the document collection
collection_path = 'COLLECTION/'

def getDocumentList(path):
    return os.listdir(path)
    
#generate document name list
doc_list = getDocumentList(collection_path)


# Load queries
queries_path = 'topics/'

#generate query name list
queries_list = getDocumentList(queries_path)


#Utilities
def divideList(list_in, percentage, seed):
    if percentage < 0 or percentage > 1:
        print('Percentage must be between 0 and 1')
        return []
    np.random.seed(seed)
    idx = np.random.shuffle(list_in)
    if percentage == 1:
        return list_in
    i = round(len(list_in) * percentage)
    return list_in[0:i]


#Processing text
#Load a stopword list NLTK
gist_file = open("stopwords.txt", "r")
try:
    content = gist_file.read()
    stopwords = content.split(",")
finally:
    gist_file.close()


# Text Processing Function
#Process text to meet the IR requirements
use_lemmatization = True

# Lemmatization
# https://www.analyticsvidhya.com/blog/2019/08
#/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python
#/?utm_source=blog&utm_medium=information-retrieval-using-word2vec-based-vector-space-model
nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
nlp.max_length=5000000

def lemmatize(x):
    return ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)])
    
    
#Returns a list of relevant terms
def processText(text):
    #remove punctuation and stopwords
    #remove single punctuation characters, remove points (not separated from string), lower case all 
    if not isinstance(text, str) :
        return []
    #remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip()
    #remove unnecessary whitespace
    text = text.replace("  ", " ")
    #lower the text
    text = text.lower()
    #lemmatize
    if use_lemmatization:
        text = lemmatize(text)
    return list(filter(lambda el: el not in stopwords, text.split()))


#Class for XML documents

class XmlFile:
    def __init__(self, xml, xml_id, field_dict, processText):
        self.field_dict = field_dict
        tree = ET.parse(xml)
        root = tree.getroot()
        for id in root.iter(xml_id):
            self.id = id.text if id.text is not None else ''
        for xml_name, field_name in field_dict.items():
            for content in root.iter(xml_name):
                self.__dict__[field_name] = processText(content.text) if content.text is not None else []
                
    def __getTF__(self, processed_text):
        #f/max_occurance of most frequent term
        num_terms = len(processed_text)
        tf_index = {}
        max_term_count = 0
        for term in processed_text:
            if term not in tf_index: #save some time
                term_count = processed_text.count(term)
                if term_count > max_term_count:
                    max_term_count = term_count
                tf_index[term] = term_count/num_terms
        #Normalization does not affect the results in this collection
        #for term in tf_index.keys():
            #tf_index[term] = tf_index[term] / max_term_count
        return tf_index


#relevant document tags 
xml_doc_id = 'DOCID'
xml_doc_fields = {'HEADLINE' : 'processed_head', 
                'TEXT' : 'processed_body'}

class XmlDoc(XmlFile):
    def __init__(self, xml):
        super().__init__(xml, xml_doc_id, xml_doc_fields, processText)       
        self.tf_head = self.__getTF__(self.processed_head)
        self.tf_body = self.__getTF__(self.processed_body)
        self.tf_text = self.__getTF__(self.processed_head + self.processed_body)


#Document Collection
class DocumentCollection:
    def __init__(self, path, doc_list, 
                 model_structure_file_path = 'model_structures/'
                ):
        self.docs_filename = model_structure_file_path+'document_collection.docs'
        self.index_filename = model_structure_file_path+'document_collection.inverted_index'
        self.global_ft_filename = model_structure_file_path+'document_collection.global_tf'
        self.total_terms_in_collection_filename = model_structure_file_path+'document_collection.total_terms_in_collection'
        self.idf_filename = model_structure_file_path+'document_collection.idf'
        self.avgdl_filename = model_structure_file_path+'document_collection.avgdl'
        self.path = path
        self.division_factor = 10
        #Inverted index 
            
    def loadFromFile(self):
        #Docs
        self.docs = {}
        for i in range(self.division_factor):
            with open(self.docs_filename+'.part_'+str(i), 'rb') as dictionary_file:
                self.docs.update(pickle.load(dictionary_file))
        #Inverted Index
        with open(self.index_filename, 'rb') as dictionary_file:
            self.inverted_index = pickle.load(dictionary_file)
        #Global TF
        with open(self.global_ft_filename, 'rb') as dictionary_file:
            self.global_tf = pickle.load(dictionary_file)
        #Idf
        with open(self.idf_filename, 'rb') as dictionary_file:
            self.idf = pickle.load(dictionary_file)
        #total terms in collection
        with open(self.total_terms_in_collection_filename, 'rb') as dictionary_file:
            self.total_terms_in_collection = pickle.load(dictionary_file)
        #Avg Doc Length
        with open(self.avgdl_filename, 'rb') as dictionary_file:
            self.avgdl = pickle.load(dictionary_file)
    
    def computeAndDump(self, doc_list):
        division_lenght = round(len(doc_list)/self.division_factor)
        #Docs
        self.docs = {}
        for i in range(self.division_factor):
            docs = {}
            st = i*division_lenght
            en = (i+1)*division_lenght
            for d in doc_list[st:en]:
                xml_document = XmlDoc(self.path + '/' + d)
                docs[xml_document.id] = xml_document
            with open(self.docs_filename+'.part_'+str(i), 'wb') as dictionary_file:
                pickle.dump(docs, dictionary_file)
            self.docs.update(docs)
        
        self.global_tf = {}
        self.total_terms_in_collection = 0
        self.inverted_index = {}
        for id in self.docs.keys():
            for term in self.docs[id].processed_head + self.docs[id].processed_body:
                self.total_terms_in_collection += 1
                if term not in self.global_tf:
                    self.global_tf[term] = 1
                else: self.global_tf[term] += 1
                if term in self.inverted_index:
                    self.inverted_index[term].add(id)
                else: self.inverted_index[term] = {id}
        for term in self.global_tf.keys():
            self.global_tf[term] = self.global_tf[term] / self.total_terms_in_collection
        
        #Table of IDF
        self.idf = {}
        for term in self.inverted_index.keys():
            self.idf[term] = np.log(len(self.docs) / len(self.inverted_index[term]))
            
        #Average document length
        self.avgdl = np.mean([len(item.processed_head + item.processed_body) for k, item in self.docs.items()])
        
        with open(self.index_filename, 'wb') as dictionary_file:
            pickle.dump(self.inverted_index, dictionary_file)
            
        with open(self.global_ft_filename, 'wb') as dictionary_file:
            pickle.dump(self.global_tf, dictionary_file)
            
        with open(self.idf_filename, 'wb') as dictionary_file:
            pickle.dump(self.idf, dictionary_file)
            
        with open(self.total_terms_in_collection_filename, 'wb') as dictionary_file:
            pickle.dump(self.total_terms_in_collection, dictionary_file)
            
        with open(self.avgdl_filename, 'wb') as dictionary_file:
            pickle.dump(self.avgdl, dictionary_file)
        
    
    def getRelevance(self, document_id, term):
        try:
            return self.docs[document_id].tf_text[term] * self.idf[term]
        except: 
            return 0


#Classes for handle XML queries
#relevant query tags 
xml_query_id = 'QUERYID'
xml_query_fields = {'TITLE' : 'processed_text'}

class XmlQuery(XmlFile):
    def __init__(self, xml):
        super().__init__(xml, xml_query_id, xml_query_fields, processText)


#Query collection
class QueryCollection:
    def __init__(self, path, query_list, filename='model_structures/query-collection.dictionary'):
        #List of XmlQuery object
        self.queries = {}
        for d in query_list:
            xml_query = XmlQuery(path + '/' + d)
            self.queries[xml_query.id] = xml_query


#Create Document Collection and Query Collection objects
recompute_all = False
#seed for random number
seed = 1
doc_list = getDocumentList(collection_path)
queries_list = getDocumentList(queries_path)
#divide the document collection (get pd% of documents)
pd = 1
doc_list = divideList(doc_list, pd, seed)

#divide the query collection (get pq% of documents)
pq = 1
queries_list = divideList(queries_list, pq, seed)

#Compute or read document collection
compute_document_collection = False or recompute_all

#create a Document collection object
document_collection = DocumentCollection(collection_path, doc_list)
if compute_document_collection:
    start = time.time()
    document_collection.computeAndDump(doc_list)
    end = time.time()
    print('Document Collection Computed in ' + str(round(end - start, 4)) + 's')
else:
    start = time.time()
    document_collection.loadFromFile()
    end = time.time()
    print('Document Collection Loaded in ' + str(round(end - start, 4)) + 's')

#Compute or read query collection
compute_query_collection = True or recompute_all
query_collection_filename = 'model_structures/query-collection.dictionary'

start = time.time()
query_collection = QueryCollection(queries_path, queries_list, query_collection_filename)
end = time.time()
print('Query Collection Computed in ' + str(round(end - start, 4)) + 's')

#Classes for ranking
class RankResult:
    def __init__(self, q_id, d_id, relevance):
        self.q_id = q_id
        self.d_id = d_id
        self.relevance = relevance
        
class RankingModel:
    def __init__(self, document_collection, query_collection, 
                 track_id='-', run_id='Rank', out_path='IR_output/'):
        self.document_collection = document_collection
        self.query_collection = query_collection
        self.track_id = track_id
        self.run_id = run_id
        self.out_path = out_path
    
    def getQueryResult(self, query, limit_result):
        rank = []
        search_space = set()
        for term in query.processed_text:
            if term in self.document_collection.inverted_index:
                search_space = search_space.union(set(self.document_collection.inverted_index[term]))
        for doc_id in search_space:
            rank += [RankResult(query.id, doc_id, self.calculateRelevance(self.document_collection.docs[doc_id], query))] 
        rank.sort(key=lambda x: x.relevance, reverse=True)
        res = ''
        for i in range(len(rank[0:limit_result])):
            res += rank[i].q_id + ' ' + self.track_id + ' ' + rank[i].d_id + ' ' + str(i) + ' ' + str(rank[i].relevance) + ' ' + self.run_id + '\n'
        return res
    
    def getReport(self, out_folder = 'Rank/', limit_result = 1000, model_name='Rank'):
        file = open(self.out_path + out_folder + self.run_id + ".out", "w")
        start = time.time()
        for q in self.query_collection.queries.keys():
            file.write(self.getQueryResult(self.query_collection.queries[q], limit_result))
        end = time.time()
        print(model_name + ' Ranking Computation time: ' + str(round(end-start, 4)) + 's')
        file.close()
    
    def calculateRelevance(self, document, query):
        return 0


#Vector space model VSM
class VectorSpaceModel(RankingModel):
    def __init__(self, document_collection, query_collection, 
                 run_id='VSM', track_id='-'
                ):
        super().__init__(document_collection, query_collection, track_id, run_id)
        
    
    def vectorizeXmlQuery(self, xml_query):
        return [self.getQueryRelevance(xml_query, t) for t in xml_query.processed_text]
    
    def vectorizeDocument(self, document, query):
        return [self.getDocumentRelevance(document, t) for t in query.processed_text]

    #tf-idf
    def getDocumentRelevance(self, document, term):
        return self.document_collection.getRelevance(document.id, term)
    def getQueryRelevance(self, query, term):
        return query.processed_text.count(term)/len(query.processed_text)
    
    def cosineSimilarity(self, vect1, vect2):
        norm1 = np.linalg.norm(vect1)
        norm2 = np.linalg.norm(vect2)
        dot_p = np.dot(vect1, vect2)
        #origin vector is equal to itself
        if dot_p == 0 and norm1 == 0 and norm2 == 0:
            return 1
        #origin vector is not equal to any vector but itself
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_p / (norm1 * norm2)

    def dotSimilarity(self, vect1, vect2):
        return np.dot(vect1, vect2)
    
    def calculateRelevance(self, document, query, similarity_function=lambda v1, v2: np.dot(v1, v2)):
        v_query = self.vectorizeXmlQuery(query)
        v_doc = self.vectorizeDocument(document, query)        
        return similarity_function(v_doc, v_query)


#Generate the VSM report for trec_eval
vsm = VectorSpaceModel(document_collection, query_collection)
vsm.getReport(out_folder = 'VSM/', limit_result = 1000, model_name='VSM')


#BM25 ranking
class BM25(VectorSpaceModel):
    def __init__(self, 
                 document_collection, 
                 query_collection, 
                 k = 1.2, 
                 b = .75,
                 track_id = '-', 
                 run_id='BM25'                 
                ):
        super().__init__(document_collection, query_collection, track_id=track_id, run_id=run_id)
        self.k = k
        self.b = b
    
    def getDocumentRelevance(self, document, term):
        try:
            idf_term = self.document_collection.idf[term]
            #x = number of occurencies of term in document
            x = document.tf_text[term] * len(document.processed_body+document.processed_head)
        except:
            return 0
        doc_len = len(document.processed_head + document.processed_body)
        normalizer = 1 - self.b + (self.b * doc_len / self.document_collection.avgdl)
        return (self.k + 1) * x / (x + self.k * normalizer) * idf_term


#Generate the BM25 report for trec_eval
bm25 = BM25(document_collection, query_collection)
bm25.getReport(out_folder = 'BM25/', limit_result = 1000, model_name='BM25')


#BM25F
class BM25F(VectorSpaceModel):
    def __init__(self, document_collection, query_collection, k=1.2, b=.75, track_id='-', run_id='BM25F',
                w_head=3, w_body=1):
        super().__init__(document_collection, query_collection, track_id=track_id, run_id=run_id)
        self.w_head = w_head
        self.w_body = w_body
        self.k = k
        self.b = b
    
    def getDocumentRelevance(self, document, term):
        try:
            idf_term = self.document_collection.idf[term]
        except:
            return 0
        x_head = document.tf_head[term] * len(document.processed_head) * self.w_head if term in document.tf_head else 0
        x_body = document.tf_body[term] * len(document.processed_body) * self.w_body if term in document.tf_body else 0
        x = x_head + x_body
        
        doc_len = len(document.processed_head + document.processed_body)
        normalizer = 1 - self.b + (self.b * doc_len / self.document_collection.avgdl)
        return (self.k + 1) * x / (x + self.k * normalizer) * idf_term
    


#Generate the BM25F report for trec_eval
bm25f = BM25F(document_collection, query_collection, w_head=3, w_body=1)
bm25f.getReport(out_folder = 'BM25F/', limit_result = 1000, model_name='BM25F')


#Unigram Language Model
class UnigramLanguageModel(RankingModel):
    def __init__(self, document_collection, query_collection, track_id='-', run_id='ULM'):
        super().__init__(document_collection, query_collection, track_id, run_id)
    
    def calculateRelevance(self, document, query):
        relevance = 0
        d_len = len(document.processed_head + document.processed_body)
        alpha = len([t for t in document.processed_head + document.processed_body if t not in query.processed_text])
        for t in query.processed_text:
            #must multiply * d_len because of the tf_text formulation in XmlDocument class
            first_term = document.tf_text[t] * d_len if t in document.tf_text else 0
            second_term = alpha * self.document_collection.global_tf[t] if t in self.document_collection.global_tf else 0#global_tf includes |c|
            denumerator = (d_len + alpha)
            result = (first_term + second_term) / denumerator
            relevance += np.log(result) if result != 0 else 0 #if a term in a query is not present in entire document collection, relevance is -inf
        return relevance


#Generate the ULM report for trec_eval
ulm = UnigramLanguageModel(document_collection, query_collection)
ulm.getReport(out_folder = 'ULM/', limit_result = 1000, model_name='ULM')

