"""
This file is specificaly deisgned for experimentation with the model, in terms of features, classsifier parameters and models
themselves with all these details proviede as banks at the bottom of the code commented with recorded analysis pertaining to 
them.
@author: Sanket Sheth (sas6792@g.rit.edu)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import re
from sklearn.metrics import confusion_matrix,f1_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk


def experiment():
    print("----------------")
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
        hate_count.append(i[2])
        offense_count.append(i[3])
        neither_count.append(i[4])
        annotators.append(i[1])
        tweets.append(i[6].strip().lower())
        labels.append(int(i[5]))
    
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
    print("Number of offensive tweets(Train): ",one)
    print("Number of Neither tweets(Train): ",two)
    print("Number of Hateful tweets(Train): ",zero)
    
    tweetsV=[]
    labelsV=[]
    annotatorsV=[]
    hate_countV=[]
    offense_countV=[]
    neither_countV=[]
    
    
    for i in validate:
        hate_countV.append(i[2])
        offense_countV.append(i[3])
        neither_countV.append(i[4])
        annotatorsV.append(i[1])
        tweetsV.append(i[6].strip().lower())
        labelsV.append(int(i[5]))
    
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
    print("Number of offensive tweets(Test): ",one)
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
    
    
    train_features=[[float(url[i]),float(retweets[i]),float(url[i]),float(atthe[i]),float(posf[i]),float(cardf[i]),float(wrd[i]),float(len_tweet[i])] for i in range(len(labels))] 
    validation_features=[[float(urlV[i]),float(retweetsV[i]),float(urlV[i]),float(attheV[i]),float(posfV[i]),float(cardV[i]),float(wrdV[i]),float(len_tweetV[i])] for i in range(len(labelsV))]
    logreg=svm.SVC()    
    logreg.fit(train_features, train_y)    
    print("Training accuraacy: ",logreg.score(train_features, train_y)*100)
    print("Testing accuracy: ",logreg.score(validation_features, validate_y)*100)
    prediction = logreg.predict(validation_features)
    print("F1 Score(Macro): ",f1_score(validate_y,prediction,average='macro')*100)
    print("Confusion matrix:")
    print(confusion_matrix(validate_y, prediction))

experiment()

    #------------------------
    #Model bank
    #logreg = linear_model.LogisticRegression()
    #logreg = RandomForestClassifier(max_depth=60, random_state=0)
    #logreg = linear_model.LogisticRegression(C=0.5,penalty="l2")
    #-----------------------

    #--------------------------------
    #Feature Bank
    #float(url[i]),   Negative
    #float(annotators[i]),   Neutral
    #float(retweets[i]), float(retweetsV[i]), Neutral
    #word numbers Very Positive
    #pos tags Very positive
    # mentions very positives specially for hate
    # cardinal numbers(hashtags count) important for neither
    #Length of tweet(Characters)  Quite useful overall for hate and neither
    
    #BAG OF WORDS - BAD BAD BAD  results
    #vectorizer=CountVectorizer()
    #train_features=vectorizer.fit_transform(tweets)
    #validation_features=vectorizer.transform(tweetsV)
    
    #-------------------------------
    
    #----------------------
    #classifier bank
    #Random Forest at depth 5 is worst
    #at depth 20 gets much much better also starts classifying hate
    #at depth 30-60 fairly similar minor changes best
    #Logistic regression normal very bad only offensive 
    #With cost at 0.5 and l1 regularizer gets a bit better but still very offensive biased
    #No change with cost a 0.9
    #No change with cost 2.5
    #No change with cost at 20 hence no effect with increased cost value
    #No change with l2 regularizer epic fail
    #felt like it took the most time
    #Good results with neither and offensive
    #----------------------
