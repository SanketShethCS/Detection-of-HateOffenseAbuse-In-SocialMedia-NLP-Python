"""
This is the main file for the project with the training and testing functionalities etched out and a pickled model ready
for use if required. This file uses a dataset provided to develop a model that classifies hateful ,offensive and Normal
tweets
@author: Sanket Sheth (sas6792@g.rit.edu)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.ensemble import RandomForestClassifier
import nltk
import pickle



def preprocess():
    dfNew= pd.read_csv('labeled_data.csv', index_col='TID', names=['TID', 'Count', 'Hate', 'Offensive', 'Neither', 'Label','Tweet'])    
    mylist=(dfNew.reset_index()[['TID', 'Count', 'Hate', 'Offensive', 'Neither', 'Label','Tweet']].values.astype(str).tolist())
    train, validate = train_test_split(mylist, test_size = 0.2, random_state = 20171206)
    return train,validate

def  train(train):
    '''
    This function trains the data on a classification model of random forest that classifies hateful, offensive and normal
    tweets.
    '''
    tweets=[]
    labels=[]
    annotators=[]
    hate_count=[]
    offense_count=[]
    neither_count=[]    
    for i in train: #for all data sample in training data
        hate_count.append(i[2])
        offense_count.append(i[3])
        neither_count.append(i[4])
        annotators.append(i[1]) #Number of annotations by humans
        tweets.append(i[6].strip().lower())
        labels.append(int(i[5])) #True training labels
    retweets=[]
    rt=re.compile(r'(\brt\b)') #To find presence of a retweet
    at=re.compile(r'(@)') #To find the number of mentions
    x=[]
    y=[]
    atthe=[]
    url=[]
    posf=[]
    wrd=[]
    cardf=[]
    posf=[]
    wrd=[]
    cardf=[]
    #Feature extraction for each tweet in training data
    for line in tweets:
        count=0
        w1=0
        card=0
        x=rt.findall(line) #Retweet feature
        y=at.findall(line) #Mention feature
        atthe.append(len(y))
        retweets.append(len(x))
        ans=line.find('http') #Presence of url in the tweet
        if (ans!=-1):
                url.append(1)
        else:
                url.append(0)
        line=line.split() #For each word in the line
        ps=nltk.pos_tag(line) #POS tagging for each word
        for w in ps:
            if w[1]=="JJ" or w[1] == "JJS" or w[1] == "JJ": #Number of adjectives in each tweet
                count=count+1
            elif w[1]=="CD": #Finding cardinal numbers
                card =card+1
            w1=w1+1 #Number of words in the tweet
        count=int(count/w1) #Nomrlaizing the adjectives in the tweet
        posf.append(count) 
        wrd.append(w1)
        cardf.append(card) 
        len_tweet=[len(i) for i in tweets] #Length of the tweet number of characters
    train_y=list(labels) #True labels of the training data
    #Feature vector
    train_features=[[float(url[i]),float(retweets[i]),float(atthe[i]),float(posf[i]),float(cardf[i]),float(wrd[i]),float(len_tweet[i])] for i in range(len(labels))]
    logreg = RandomForestClassifier(max_depth=60, random_state=0) #Initializing the model
    logreg.fit(train_features, train_y) #Training the model
    print("Classificaation rate(Train): ",logreg.score(train_features, train_y)*100) #Classification rate for training data
    pickle.dump(logreg,open("Project_Classifier_model.sav","wb")) #Saving the training model for future use
    
def predict(validate):
    '''
    This function is basically the testing fucntion of the model that predicts the classification
    labels for the testing data and gives the confusion matrix along with the F1 Measure for macro average.
    '''
    logreg=pickle.load(open("Project_Classifier_model.sav","rb")) #Using the pickled model for testing
    tweetsV=[]
    labelsV=[]
    annotatorsV=[]
    hate_countV=[]
    offense_countV=[]
    neither_countV=[]    
    for i in validate: #For each sample in test data
        hate_countV.append(i[2])    
        offense_countV.append(i[3])
        neither_countV.append(i[4])
        annotatorsV.append(i[1]) #number of annotators for each data sample
        tweetsV.append(i[6].strip().lower())
        labelsV.append(int(i[5]))    #True labels assigned by annotations from the dataset
    retweetsV=[]
    rt=re.compile(r'(\brt\b)') #To detect whether it is a retweet or not
    at=re.compile(r'(@)') #To find the number of mentions in the tweets
    x=[]
    y=[]
    attheV=[]
    posfV=[]
    wrdV=[]
    cardV=[]
    urlV=[]
    
    for line in tweetsV: #for each tweet feature extraction
        count=0
        card=0
        w1=0
        x=rt.findall(line) #Finding retweet feature
        y=at.findall(line) #Finding mention feature
        attheV.append(len(y))
        retweetsV.append(len(x))
        ans=line.find('http') #Finding url feature
        if (ans!=-1):
            urlV.append(1)
        else:
            urlV.append(0)
        line=line.split() #For each word in the tweet
        ps=nltk.pos_tag(line) #POS Tagging using nltk
        for w in ps: 
            if w[1]=="JJ" or w[1] == "JJS" or w[1] == "JJ": #Number of adjectives
                count=count+1
            elif w[1] == "CD": #Finding cardinal numbers
                card=card+1
            w1=w1+1 #Number of words in the tweet
        count=int(count/w1) #Normalising the adjective count
        posfV.append(count) 
        wrdV.append(w1)
        cardV.append(card)
        len_tweetV=[len(i) for i in tweetsV] #Number of characters in the tweet
    validate_y=list(labelsV) #True labels        
    #Feature vector for each tweet
    validation_features=[[float(urlV[i]),float(retweetsV[i]),float(attheV[i]),float(posfV[i]),float(cardV[i]),float(wrdV[i]),float(len_tweetV[i])] for i in range(len(labelsV))]       
    prediction = logreg.predict(validation_features) #Predicting labels for training data
    print("F1 Score(Macro): ",f1_score(validate_y,prediction,average='macro')*100) #F1 score for macro average
    print("Confusion Matrix: ")
    print(confusion_matrix(validate_y, prediction)) #Confusion Matrix for testing data
    return validation_features,validate_y


def score(validate_y,validation_features):  
    '''
    This function uses the saved model to find out the classification rate for the testing data 
    '''
    logreg=pickle.load(open("Project_Classifier_model.sav","rb")) #Loading pickled model
    print("Classification rate(Test): ",logreg.score(validation_features, validate_y)*100) #Printing the classification rate

def main():
    '''
    This is the main function that controls the structure of the system.
    '''
    print("----------------")
    traindata,testdata=preprocess()  #Preprocess the data i.e extract, clean and split
    #train(traindata) #Uncomment this if required to train on data for changes if made any
    features,testlabels=predict(testdata) #This tests the model on testing data
    score(testlabels,features) #Gives the classification rate
    print("----------------")


main()



