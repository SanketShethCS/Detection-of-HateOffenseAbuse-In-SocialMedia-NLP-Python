"""
This file is specifically designed to compare the random classifier model with just a binary classification between Neither 
and hate tweets also with added functionality of normalizing the data commented inside the code for both training and validation.
@author: Sanket Sheth (sas6792@g.rit.edu)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.ensemble import RandomForestClassifier

print("For 0/2")
df = pd.read_csv('labeled_data.csv')
dfNew= pd.read_csv('labeled_data.csv', index_col='TID', names=['TID', 'Count', 'Hate', 'Offensive', 'Neither', 'Label','Tweet'])

mylist=(dfNew.reset_index()[['TID', 'Count', 'Hate', 'Offensive', 'Neither', 'Label','Tweet']].values.astype(str).tolist())
train, validate = train_test_split(mylist, test_size = 0.2, random_state = 20171206)

tweets=[]
labels=[]
annotators=[]
hate_count=[]
offense_count=[]
neither_count=[]

for i in train:
    if i[5]!='1':
        hate_count.append(i[2])
        offense_count.append(i[3])
        neither_count.append(i[4])
        annotators.append(i[1])
        tweets.append(i[6].strip().lower())
        labels.append(int(i[5]))
        
#--------------------------------------
#Normalising the data set
#count=0
#c1=0
#c2=0
#for i in train:
#    if count == 2286:
#        break
#    if i[5]=='2'and c1 <1143:
#        hate_count.append(i[2])
#        offense_count.append(i[3])
#        neither_count.append(i[4])
#        annotators.append(i[1])
#        tweets.append(i[6].strip().lower())
#        labels.append(int(i[5]))
#        count=count+1
#        c1=c1+1
#    elif i[5]=='0' and c2<=1143:
#        hate_count.append(i[2])
#        offense_count.append(i[3])
#        neither_count.append(i[4])
#        annotators.append(i[1])
#        tweets.append(i[6].strip().lower())
#        labels.append(int(i[5]))
#        count=count+1
#        c2==c2+1
#--------------------------------------


one=0
two=0
zero=0
for i in labels:
    if i == 1:
        one +=1
    elif i==2:
        two+=1
    else:
        zero+=1
print("Number of Neither tweets(Train): ",two)
print("Number of Hateful tweets(Train): ",zero)
    
print("---------------------")

tweetsV=[]
labelsV=[]
annotatorsV=[]
hate_countV=[]
offense_countV=[]
neither_countV=[]


for i in validate:
    if i[5]!='1':
        hate_countV.append(i[2])
        offense_countV.append(i[3])
        neither_countV.append(i[4])
        annotatorsV.append(i[1])
        tweetsV.append(i[6].strip().lower())
        labelsV.append(int(i[5]))


#--------------------------------------
#Normalising the data set
#count=0
#c1=0
#c2=0
#for i in validate:
#    if count == 574:
#        break
#    if i[5]=='2'and c1<287: 
#        hate_countV.append(i[2])
#        offense_countV.append(i[3])
#        neither_countV.append(i[4])
#        annotatorsV.append(i[1])
#        tweetsV.append(i[6].strip().lower())
#        labelsV.append(int(i[5]))
#        count=count+1
#        c1=c1+1
#    elif i[5]=='0' and c2<=287:
#        hate_countV.append(i[2])
#        offense_countV.append(i[3])
#        neither_countV.append(i[4])
#        annotatorsV.append(i[1])
#        tweetsV.append(i[6].strip().lower())
#        labelsV.append(int(i[5]))
#        count=count+1
#        c2=c2+1
#--------------------------------------

one=0
two=0
zero=0
for i in labelsV:
    if i == 1:
        one +=1
    elif i==2:
        two+=1
    else:
        zero+=1
print("Number of Neither tweets(Test): ",two)
print("Number of Hateful tweets(Test): ",zero)



retweets=[]
retweetsV=[]
rt=re.compile(r'(\brt\b)')
at=re.compile(r'(@)')
x=[]
y=[]
atthe=[]
attheV=[]
for i in tweets:
    x=rt.findall(i)
    y=at.findall(i)
    atthe.append(len(y))
    retweets.append(len(x))
for i in tweetsV:
    x=rt.findall(i)
    y=at.findall(i)
    attheV.append(len(y))
    retweetsV.append(len(x))

import nltk
posf=[]
wrd=[]
cardf=[]
for line in tweets:
    count=0
    w1=0
    card=0
    line=line.split()
    ps=nltk.pos_tag(line)
    for w in ps:
        if w[1]=="JJ" or w[1] == "JJS" or w[1] == "JJ":
            count=count+1
        elif w[1]=="CD":
            card =card+1
        w1=w1+1
    count=int(count/w1)
    posf.append(count)
    wrd.append(w1)
    cardf.append(card)
    
posfV=[]
wrdV=[]
cardV=[]
for line in tweetsV:
    count=0
    card=0
    w1=0
    line=line.split()
    ps=nltk.pos_tag(line)
    for w in ps:
        if w[1]=="JJ" or w[1] == "JJS" or w[1] == "JJ":
            count=count+1
        elif w[1] == "CD":
            card=card+1
        w1=w1+1
    count=int(count/w1)
    posfV.append(count)
    wrdV.append(w1)
    cardV.append(card)

len_tweet=[len(i) for i in tweets]
len_tweetV=[len(i) for i in tweetsV]

validate_y=list(labelsV)
train_y=list(labels)


url=[]
for i in tweets:
    ans=i.find('http')
    if (ans!=-1):
        url.append(1)
    else:
        url.append(0)
urlV=[]
for i in tweetsV:
    ans=i.find('http')
    if (ans!=-1):
        urlV.append(1)
    else:
        urlV.append(0)

train_features=[[float(atthe[i]),float(posf[i]),float(cardf[i]),float(wrd[i]),float(len_tweet[i])] for i in range(len(labels))] 
validation_features=[[float(attheV[i]),float(posfV[i]),float(cardV[i]),float(wrdV[i]),float(len_tweetV[i])] for i in range(len(labelsV))]
logreg = RandomForestClassifier(max_depth=60, random_state=0)
logreg.fit(train_features, train_y)
print("Training accuraacy: ",logreg.score(train_features, train_y)*100)
print("Testing accuracy: ",logreg.score(validation_features, validate_y)*100)
prediction = logreg.predict(validation_features)
print("F1 Score(Macro): ",f1_score(validate_y,prediction,average='macro')*100)
print("Confusion matrix:")
print(confusion_matrix(validate_y, prediction))
