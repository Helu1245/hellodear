#!/bin/bash

# Install necessary packages
pip install pandas scikit-learn

# Download the compressed dataset
wget https://raw.githubusercontent.com/PacktPublishing/Mastering-Machine-Learning-for-Penetration-Testing/master/Chapter03/Android_Feats.csv

wget https://raw.githubusercontent.com/PacktPublishing/Mastering-Machine-Learning-for-Penetration-Testing/master/Chapter03/MalwareData.csv.gz

# Extract the dataset
gunzip -d MalwareData.csv.gz

# Python script starts here
python3 <<EOF
# In[5]:


import pandas as pd
MalwareDataset = pd.read_csv('MalwareData.csv', sep='|')
Legit = MalwareDataset[0:41323].drop(['legitimate'], axis=1)
Malware = MalwareDataset[41323::].drop(['legitimate'], axis=1)


# In[6]:


print('The Number of important features is %i \n' % Legit.shape[1])
print("The shape of legit dataset is: %s samples, %s features"%(Legit.shape[0],Legit.shape[1]))
print("The shape of malware dataset is: %s samples, %s features"%(Malware.shape[0],Malware.shape[1]))


# In[7]:


print(MalwareDataset.columns)


# In[8]:


print(MalwareDataset.head(5))


# In[9]:


pd.set_option("display.max_columns", None)
print(MalwareDataset.head(5))


# In[10]:


print(Legit.take([1]))


# In[11]:


print(Malware.take([1]))


# In[23]:


import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split


# In[24]:


Data = MalwareDataset.drop(['Name', 'md5', 'legitimate'], axis=1).values
Target = MalwareDataset['legitimate'].values
FeatSelect = sklearn.ensemble.ExtraTreesClassifier().fit(Data, Target)
Model = SelectFromModel(FeatSelect, prefit=True)
Data_new = Model.transform(Data)
print (Data.shape)
print (Data_new.shape)


# In[25]:


import numpy as np
Features = Data_new.shape[1]
importances = FeatSelect.feature_importances_
Index = np.argsort(importances)[::-1]
for feat in range(Features):
    print("%d"%(feat+1),MalwareDataset.columns[2+Index[feat]],importances[Index[feat]])


# In[26]:


from sklearn.ensemble import RandomForestClassifier
Legit_Train, Legit_Test, Malware_Train, Malware_Test = train_test_split(Data_new, Target ,test_size=0.2)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(Legit_Train, Malware_Train)
score = clf.score(Legit_Test, Malware_Test)


# In[27]:


print("The score of Random Forest Algorithm is:",score*100)


# In[28]:


from sklearn.metrics import confusion_matrix
Result = clf.predict(Legit_Test)
CM = confusion_matrix(Malware_Test, Result)


# In[29]:


CM.shape


# In[32]:


type(CM)


# In[34]:


CM


# In[33]:


print("False positive rate : %f %%" % ((CM[0][1] / float(sum(CM[0])))*100))
print('False negative rate : %f %%' % ( (CM[1][0] /float(sum(CM[1]))*100)))


# In[42]:


Clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=50)
Clf.fit(Legit_Train, Malware_Train)
Score = Clf.score(Legit_Test, Malware_Test)


# In[43]:


print ("The Model score using Gradient Boosting is", Score * 100)


# In[45]:


Classifiers = { "RandomForest": sklearn.ensemble.RandomForestClassifier(n_estimators=50),"GradientBoosting": sklearn.ensemble.GradientBoostingClassifier(n_estimators=50),"AdaBoost": sklearn.ensemble.AdaBoostClassifier(n_estimators=100),}
for Classif in Classifiers:
    clf = Classifiers[Classif]
    clf.fit(Legit_Train,Malware_Train)
    score = clf.score(Legit_Test, Malware_Test)
    print("%s : %f %%" % (Classif, score*100))


# In[46]:


from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
import csv
import random


# In[49]:


PRatio = 0.7
Dataset = open('Android_Feats.csv')
Reader = csv.reader(Dataset)
Data = list(Reader)
Data = random.sample(Data, len(Data))
Data = np.array(Data)
Dataset.close()


# In[51]:


cols = np.shape(Data)[1]
Y = Data[:,cols-1]
Y = np.array(Y)
Y = np.ravel(Y,order='C')
X = Data[:,:cols-1]
X = X.astype(float)
X = preprocessing.scale(X)


# In[52]:


Features = [i.strip() for i in open("Android_Feats.csv").readlines()]
Features = np.array(Features)
MI= mutual_info_classif(X,Y)
Featureind = sorted(range(len(MI)), key=lambda i: MI[i], reverse=True)[:50]
SelectFeats = Features[Featureind]


# In[53]:


PRows = int(PRatio*len(Data))
TrainD = X[:PRows,Featureind]
TrainL = Y[:PRows]
TestD = X[PRows:,Featureind]
TestL = Y[PRows:]


# In[54]:


clf = svm.SVC()
clf.fit(TrainD,TrainL)
score = clf.score(TestD,TestL)
print (score * 100)
EOF

# Ask user if they want to delete the downloaded content
read -p "Do you want to delete the downloaded content? (y/n): " delete_choice
if [ "$delete_choice" == "y" ] || [ "$delete_choice" == "Y" ]; then
    rm MalwareData.csv
    rm Android_Feats.csv
fi

