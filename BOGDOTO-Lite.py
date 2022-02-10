#!/usr/bin/env python
# coding: utf-8

# In[120]:


from langdetect import detect
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing,feature_selection, metrics
from sklearn.linear_model import Ridge 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

## for explainer
from lime import lime_text
from geopy.geocoders import Nominatim


# In[67]:


df = pd.read_csv("33000-BORDEAUX_nettoye.csv")
print(df.head(3))


# In[68]:


print(df.columns)


# In[69]:


sns.set() 
sns.histplot(data=df, x="PrixNuitee", binwidth=20)


# # Nettoyage des données

# In[70]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    # Lowercase text
    sentence = sentence.lower()
    # Remove whitespace
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    # Remove weblinks
    rem_url=re.sub(r'http\S+', '',cleantext)
    # Remove numbers
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    # Remove StopWords
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('french')]
    # Use lemmatization
    # lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(filtered_words)


# In[71]:


df['Resume_pre']=df['Resume'].map(lambda s:preprocess(s)) 


# In[72]:


df['Description_pre']=df['Description'].map(lambda s:preprocess(s)) 


# ### suppression de l'anglais : pour les colonnes Resume et Description 

# In[73]:


c=0
for i in range (len(df['Resume_pre'])):
    if str(df['Resume_pre'][i])=='':
        c=c+1
        df=df.drop(i, axis=0)
df.reset_index(inplace=True, drop=True)
print(c)
print(len(df['Resume_pre']))
c=0
for i in range (len(df['Resume_pre'])):
    truc=detect(df['Resume_pre'][i])
    if str(truc)=='en':
        c=c+1
        df=df.drop(i, axis=0)
df.reset_index(inplace=True, drop=True)
print(c)
print(len(df['Resume_pre']))


# In[74]:


c=0
for i in range (len(df['Description_pre'])):
    if str(df['Description_pre'][i])=='':
        c=c+1
        df=df.drop(i, axis=0)
df.reset_index(inplace=True, drop=True)
print(c)
print(len(df['Description_pre']))
c=0
for i in range (len(df['Description_pre'])):
    truc=detect(df['Description_pre'][i])
    if str(truc)=='en':
        c=c+1
        df=df.drop(i, axis=0)
df.reset_index(inplace=True, drop=True)
print(c)
print(len(df['Description_pre']))


# ### suppression des prix valant 0€ ou >200€

# In[75]:


c=0
for i in range (len(df['PrixNuitee'])):
    if df['PrixNuitee'][i]==0 or df['PrixNuitee'][i]>200:
        c=c+1
        df=df.drop(i, axis=0)
df.reset_index(inplace=True, drop=True)
print(c)
print(len(df['PrixNuitee']))


# In[76]:


sns.set() 
sns.histplot(data=df, x="PrixNuitee", binwidth=20)


# In[77]:


f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=10);


# In[82]:


correlations=df.corr()


# In[78]:


df["PrixNuitee"].describe()

prix_classe = []

for prix in df["PrixNuitee"]:
    p = prix / 20
    prix_classe.append(int(p))

df["PrixClasse"] = prix_classe
df["PrixClasse"].describe()


# In[84]:


correlations.sort_values(by=['PrixNuitee'])


# In[79]:


df["PrixClasse"].value_counts(normalize=True).plot(kind='pie')


# # Récupération des nombres

# In[80]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)
newdf


# In[85]:


Xn=newdf.drop(["PrixClasse"], axis=1)

Xn=Xn.drop(["Shampooing"], axis=1)
Xn=Xn.drop(["PrixNuitee"], axis=1)
Xn=Xn.drop(["prix_nuitee"], axis=1)

Xn=Xn.drop(["petit_dejeuner"], axis=1)
Xn=Xn.drop(["Identifiant"], axis=1)
Xn=Xn.drop(["accessibilite"], axis=1)
Xn=Xn.drop(["animaux_acceptes"], axis=1)
Xn=Xn.drop(["porte_chambre_verrou"], axis=1)
Xn=Xn.drop(["logement_fumeur"], axis=1)
Xn=Xn.drop(["Ascenseur"], axis=1)
Xn=Xn.drop(["Entree_24-24"], axis=1)
Xn=Xn.drop(["produits_base"], axis=1)
Xn=Xn.drop(["kit_secours"], axis=1)


# In[81]:


yn = df["PrixClasse"].values


# In[90]:


pca = PCA(n_components=1)
pca.fit(Xn)
X_pca = pca.transform(Xn)

plt.scatter(X_pca, yn, c='pink')


# In[91]:


tsne = TSNE(n_components=1, random_state=0)
X_tsne = tsne.fit_transform(Xn)

plt.scatter(X_tsne, yn, c='pink')


# # récupération du texte

# In[106]:


#words on whole set
y = df["PrixNuitee"].values
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2))
#vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
corpus = df["Resume_pre"]
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X_names = vectorizer.get_feature_names()
p_value_limit = 0.98
dtf_features = pd.DataFrame()

for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[False,True])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()
print(X_names)
print(len(X_names))
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X=X.toarray()
print(X)
print(X.shape)


# In[107]:


A=X


# In[108]:


#words on whole set
y = df["PrixNuitee"].values
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2))
#vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
corpus = df["Description_pre"]
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X_names = vectorizer.get_feature_names()
p_value_limit = 0.98
dtf_features = pd.DataFrame()

for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[False,True])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()
print(X_names)
print(len(X_names))
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X=X.toarray()
print(X)
print(X.shape)


# In[109]:


B=X


# In[110]:


A=np.concatenate((A, B), axis=1)


# In[111]:


#words on whole set
y = df["PrixNuitee"].values
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2))
#vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
corpus = df["type_propriete"]
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95
dtf_features = pd.DataFrame()

for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[False,True])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()
print(X_names)
print(len(X_names))
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X=X.toarray()
print(X)
print(X.shape)


# In[112]:


C=X


# In[113]:


A=np.concatenate((A, C), axis=1)


# In[114]:


#words on whole set
y = df["PrixNuitee"].values
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2))
#vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
corpus = df["Type_logement"]
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95
dtf_features = pd.DataFrame()

for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[False,True])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()
print(X_names)
print(len(X_names))
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
X=X.toarray()
print(X)
print(X.shape)


# In[115]:


D=X


# In[116]:


A=np.concatenate((A, D), axis=1)
print(A)


# In[117]:


print(A.shape)


# # partie entrainement sur vecteur final

# In[183]:


#lier df avec valeurs de yn : 
YN=pd.DataFrame(df["PrixClasse"])
YN["PrixNuitee"]=df["PrixNuitee"]


# In[184]:


YN=YN.to_numpy()
print(YN)


# In[118]:


V_final=np.concatenate((Xn, A), axis=1)
print(V_final)
print(V_final.shape)


# In[185]:


from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(V_final, YN[:,0], test_size=0.33,random_state = 0)

bayes_ridge = linear_model.BayesianRidge()
bayes_ridge.fit(X_train, y_train)
print("Training score R^2 =", bayes_ridge.score(X_train, y_train)) 
print("Test score R^2 =", bayes_ridge.score(X_test, y_test))


# # Construction de la prédiction du jeu de tests

# In[173]:


predictions_test=[]
test=[]

c=0
for i in range (len(y_test)):
    test.append(y_test[c])
    predictions_test.append(bayes_ridge.predict([X_test[c]]))
    c=c+1


predictions_train=[]
train=[]

c=0
for i in range (len(y_train)):
    train.append(y_train[c])
    predictions_train.append(bayes_ridge.predict([X_train[c]]))
    c=c+1

predictions=np.concatenate((predictions_test,predictions_train), axis=0)
print(predictions)

attendus=np.concatenate((test,train), axis=0)
print(attendus)


# In[189]:


predictions=[]
attendus=[]

c=0
for i in range (len(YN[:,0])):
    attendus.append(YN[c,0])
    predictions.append(bayes_ridge.predict([V_final[c]]))
    c=c+1


#print(predictions)
#print(attendus)


# In[192]:


data=[]
dfprediction= pd.DataFrame(data, columns=['predictions', 'attendusClasse'])
dfprediction['predictions']=pd.DataFrame(predictions)
dfprediction
dfprediction['attendusClasse']=pd.DataFrame(attendus)
dfprediction
dfprediction['prixPredit']=dfprediction['predictions']*20
dfprediction
dfprediction['PrixNuitee']=pd.DataFrame(YN[:,1])
dfprediction


# In[193]:


dfprediction['erreur']=abs(dfprediction['PrixNuitee']-dfprediction['prixPredit'])
dfprediction


# In[ ]:


avg=dfprediction['erreur'].mean()
print(avg)


# In[ ]:




