
# coding: utf-8

# # NER with NLTK MEMM

# ## Results

#  |#ID | #Gold Standard | #Found | #Correct | Precision | Recall | F-1 |
#  |----|----------------|--------|----------|-----------|--------|-----|
#  |1|3413|45|1|0.022222222222222223|0.0002929973630237328|0.000578368999421631|
#  |2|3413|291|61|0.209621993127|0.0178728391444|0.0329373650108|
#  |3|3413|1843|629|0.341291372762|0.184295341342|0.239345509893|
#  |4|3413|1729|660|0.381723539618|0.193378259596|0.256709451575|
#  |5|3413|2830|833|0.294346289753|0.244066803399|0.266858881948|
#  |6|3413|1950|843|0.432307692308|0.246996777029|0.314376281932|
# |7|3413|2090|1009|0.482775119617|0.295634339291|0.366709067781|
# |8|3413|2217|1118|0.504285069914|0.327571051861|0.397158081705|
# |9|3413|2396|1338|0.558430717863|0.392030471726|0.460664486142|
# |10|3413|2355|1319|0.56008492569|0.386463521828|0.457350901526|
# |11|3413|2479|1432|0.577652279145|0.41957222385|0.486082824168|
# |12|3413|2601|1556|0.598231449443|0.455903896865|0.517459261723|
# |13|3372|2570|1508|0.586770428016|0.447212336892|0.507573207674|
# |14|3413|2641|1566|0.592957213177|0.458833870495|0.517343904856|
# |15|3263|2445|1420|0.580777096115|0.435182347533|0.497547302032|
# |16|3413|2698|1765|0.65418828762|0.517140345737|0.577646866307|
# |17|3413|2854|2075|0.72704975473|0.607969528274|0.662198819212|
# |18|3413|2803|1886|0.672850517303|0.552593026663|0.606821106821|
# |19|3371|2750|1812|0.658909090909|0.537525956689|0.592060120895|
# |20|3188|2496|1602|0.641826923077|0.502509410289|0.563687543983|
# |21|3395|2615|1653|0.632122370937|0.486892488954|0.550083194676|
# |22|3321|2507|1540|0.614280015955|0.463715748269|0.528483184626|
# |23|3448|2574|1600|0.621600621601|0.46403712297|0.531384921953|
# ||3388|2856|1976|0.6918767507|0.583234946871|0.632927610506|
# Training on 80% of data and 20% dev with NLTK MaxEnt
# 
#  1. Feature : POS, Word Position
#  2. Feature : POS, Word Position, Word
#  3. Feature : POS, Word Position, Word, word Shape
#  4. Feature : POS, Previous POS, Word Position, Word, word Shape
#  5. Feature : current pos, previous pos, word position, current word, previous word, word Shape
#  6. Feature : current pos, previous pos, word position, current word, previous word, current word shape, previous word shape
#  7. Feature : {current, previous, next} pos, word position, {current, previous, next} word, {current, previous, next} word shape
#  8. Feature : {current, previous, next} pos, word position, {current, previous, next} word, {current, previous, next} word shape, current word len
#  9. Feature : {current, previous, next} {pos, word, word shape}, current word len, word position, suffix3, prefix3
#  10. Feature : {current, previous, next} {pos, word, word_shape, lemma},  word position, current word len, suffix3, prefix3
#  11. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len},  word position, suffix3, prefix3
#  12. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len, suffix3, prefix3},  word position
#  13. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len, suffix3, prefix3},  word position
#      - Random Data Set
#  14. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len, suffix3, prefix3},  word position, currrent pos + previous pos, current pos + next pos
#  
#  15. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len, suffix3, prefix3},  word position, currrent pos + previous pos, current pos + next pos
#      - Random Data Set
#  16. Feature : {current, previous, next} {pos, word, word_shape, lemma, word len, suffix3, prefix3},  word position, currrent pos + previous pos, current pos + next pos, prev_tag
#  17. Same as 16. but different data set

# In[1]:


# In[2]:


import pandas as pd
import numpy as np
from StringIO import StringIO
from collections import Counter
import itertools
import nltk
from nltk import MaxentClassifier
from nltk.chunk import named_entity
from nltk.stem.snowball import SnowballStemmer


# In[3]:


text = open("./gene-trainF17.txt").read()
lines = [ y.strip() for y in text.split("\n\n")]
raw_df = pd.DataFrame(lines, columns = ["sentence"])
# msk = np.random.rand(len(raw_df)) < 0.8
# train_df = raw_df[msk]
# dev_df = raw_df[~msk]
stemmer = SnowballStemmer("english")


# ## Training Phase

# In[4]:


df = raw_df.copy()


# In[5]:


df.loc[:, "sentence_token"] = df["sentence"].apply(lambda x : tuple(y.split("\t") for y in x.split("\n")))


# In[6]:


df.loc[:, "tags"] = df["sentence_token"].apply(lambda x : tuple(y[2] for y in x))
df.loc[:, "words"] = df["sentence_token"].apply(lambda x : tuple(y[1] for y in x))


# In[7]:


df_count = df["words"].value_counts()
word_counter = Counter()

for k in df_count.keys():
    for w in k:
        word_counter[w] += 1

V = set(k for k,v in word_counter.iteritems() if v > 1)


# In[8]:


df.loc[:, "words_"] = df["sentence_token"].apply(lambda x : tuple(y[1] for y in x))
df.loc[:, "pos"] = df["words_"].apply(lambda x : tuple( y[1] for y in nltk.pos_tag(x)))
df.loc[:, "words"] = df["words_"].apply(lambda x : tuple(y for y in x))


# In[9]:


def extract_features(num, word_tuple, pos_tuple, prev_tag):
    feature = {}
    next_elem = lambda itr, idx: itr[idx + 1] if idx < len(itr) - 1 else None
    prev_elem = lambda itr, idx: itr[idx - 1] if idx > 0 else None
    
#     word_tuple = map(lambda x : x if x in V else "UNK", word_tuple)
    prev_word = prev_elem(word_tuple, num)
    word = word_tuple[num]
    next_word = next_elem(word_tuple, num)
    
    prev_pos = prev_elem(pos_tuple, num)
    pos = pos_tuple[num]
    next_pos = next_elem(pos_tuple, num)
    
    feature["index"] = num
    
    prefix = lambda x,y : x[:y].lower() if x else None
    
    feature["prev_prefix3"] = prefix(prev_word,3)
    feature["curr_prefix3"] = prefix(word,3)
    feature["next_prefix3"] = prefix(next_word,3)
    
    suffix = lambda x, y : x[-y:].lower() if x else None
    
    feature["prev_suffix3"] = suffix(prev_word,3)
    feature["curr_suffix3"] = suffix(word,3)
    feature["next_suffix3"] = suffix(next_word,3)
    
    feature["prev_len"] = len(prev_word) if prev_word else 0
    feature["curr_len"] = len(word)
    feature["next_len"] = len(next_word) if next_word else 0
    
    feature["unknown_word"] = word if word in V else "#UNK#"
    
    feature["prev_word"] = prev_word
    feature["curr_word"] = word
    feature["next_word"] = next_word
    
    feature["prev_pos"] = prev_pos
    feature["curr_pos"] = pos
    feature["next_pos"] = next_pos
    
    
    feature["curr_shape"] = named_entity.shape(word)
    feature["prev_shape"] = named_entity.shape(prev_word) if prev_word else None
    feature["next_shape"] = named_entity.shape(next_word) if next_word else None
    
    
    feature["curr_lemma"] = stemmer.stem(word)
    feature["prev_lemma"] = stemmer.stem(prev_word) if prev_word else None
    feature["next_lemma"] = stemmer.stem(next_word) if next_word else None

    feature["prev_prefix3+curr_prefix3"] = "%s+%s" % (feature["prev_prefix3"], feature["curr_prefix3"])
    feature["curr_prefix3+next_suffix3"] = "%s+%s" % (feature["curr_prefix3"], feature["next_suffix3"])
    feature["prev_prefix3+curr_prefix3+next_suffix3"] = "%s+%s+%s" % (feature["prev_prefix3"], feature["curr_prefix3"], feature["next_suffix3"])
    
    feature["prev_suffix3+curr_suffix3"] = "%s+%s" % (feature["prev_suffix3"], feature["curr_suffix3"])
    feature["curr_suffix3+next_suffix3"] = "%s+%s" % (feature["curr_suffix3"], feature["next_suffix3"])
    feature["prev_suffix3+curr_suffix3+next_suffix3"] = "%s+%s+%s" % (feature["prev_suffix3"], feature["curr_suffix3"], feature["next_suffix3"])
                                     
    feature["curr_pos+next_pos"] = "%s+%s" % (pos, next_pos)
    feature["prev_pos+curr_pos"] = "%s+%s" % (prev_pos, pos)
    feature["prev_pos+curr_pos+next_pos"] = "%s+%s+%s" %(prev_pos, pos, next_pos)
    
    feature["prev_shape+curr_shape"] = "%s+%s" % (feature["prev_shape"], feature["curr_shape"])
    feature["curr_shape+next_shape"] = "%s+%s" % (feature["curr_shape"], feature["next_shape"])
    feature["prev_shape+curr_shape+next_shape"] = "%s+%s+%s" % (feature["prev_shape"], feature["curr_shape"], feature["next_shape"])
    
    feature["prev_lemma+curr_lemma"] = "%s+%s" % (feature["prev_lemma"], feature["curr_lemma"])
    feature["curr_lemma+next_lemma"] = "%s+%s" % (feature["curr_lemma"], feature["next_lemma"])
    feature["prev_lemma+curr_lemma+next_lemma"] = "%s+%s+%s" % (feature["prev_lemma"], feature["curr_lemma"], feature["next_lemma"])
    
    feature["curr_word+next_word"] = "%s+%s" % (word, next_word)
    feature["prev_word+curr_word"] = "%s+%s" % (prev_word, word)
    feature["prev_word+curr_word+next_word"] = "%s+%s+%s" %(prev_word, word, next_word)
    
    feature["prev_tag1"]  = prev_tag[0]
    feature["prev_tag2"]  = prev_tag[1]
    feature["prev_tag3"]  = prev_tag[2]
    return feature




# In[10]:


features = []
for index, tuples in df[["words", "pos", "tags"]].iterrows():
    word_tuple, pos_tuple, tag_tuple = tuples
    word_num = 0
    prev_tag = prev_tag1 = prev_tag2 = None
    for word_num in range(len(word_tuple)):
        feature = (extract_features(word_num, word_tuple, pos_tuple, [prev_tag2, prev_tag1, prev_tag]), tag_tuple[word_num])
        features.append(feature)
        prev_tag = tag_tuple[word_num]
        prev_tag1 = prev_tag
        prev_tag2 = prev_tag1
    


# In[11]:


memm_classifier = MaxentClassifier.train(features, "megam")


# ## Testing Phase

# In[12]:


text = open("./goldoutput.txt").read()
lines = [ y.strip() for y in text.split("\n\n")]
test_df = pd.DataFrame(lines, columns = ["sentence"])
# test_df = dev_df.copy()
test_df.loc[:, "sentence_token"] = test_df["sentence"].apply(lambda x : tuple(y.split("\t") for y in x.split("\n")))
test_df.loc[:, "words_"] = test_df["sentence_token"].apply(lambda x : tuple(y[1] for y in x))
test_df.loc[:, "pos"] = test_df["words_"].apply(lambda x :  tuple(x[1] for x in nltk.pos_tag(x)))
test_df.loc[:, "words"] = test_df["words_"].apply(lambda x : tuple(y for y in x))


# In[13]:


predictions = []
for _, tuples in test_df[["pos", "words"]].iterrows():
    pos_tuple, word_tuple = tuples
    prev_tag = prev_tag1 = prev_tag2 = None
    prediction = []
    for word_num in range(len(word_tuple)):
        feature = extract_features(word_num, word_tuple, pos_tuple, [prev_tag2, prev_tag1, prev_tag])
        prev_tag = predict = memm_classifier.classify(feature)
        prev_tag1 = prev_tag
        prev_tag2 = prev_tag1
        prediction.append(predict)
    predictions.append(prediction)


# In[14]:


test_df["prediction"] = predictions


# In[15]:


test_df.loc[:, "temp1"] = test_df[["prediction", "words_"]].apply(lambda x : [str(i) + "\t" + "\t".join(y) for i,y in enumerate(zip( x["words_"], x["prediction"]), 1)], axis = 1)
test_df.loc[:, "temp1"] = test_df["temp1"].apply(lambda x : "\n".join(x))

predictions = "\n\n".join(test_df["temp1"].tolist())
# gold_standard = "\n\n".join(test_df["sentence"].tolist())
# eval(StringIO(gold_standard), StringIO(predictions))


# In[16]:


with open("sharma-hasil-assgn4-out.txt", "w") as f:
    f.write(predictions)
    f.write("\n")

