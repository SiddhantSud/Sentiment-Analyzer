from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas
from newsapi import NewsApiClient


def get_all_data():
    root = "Data/"

    with open(root +"imdb_labelled.txt","r") as text_file:
        data = text_file.read().split('\n')

    with open(root +"amazon_cells_labelled.txt","r") as text_file:
        data += text_file.read().split('\n')

    with open(root +"yelp_labelled.txt","r") as text_file:
        data += text_file.read().split('\n')

    return data


def preprocessing_data(data):
    processing_data =[]
    for single_data in data:
        if len(single_data.split("\t"))== 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))

    return processing_data

def split_data(data):
    total=len (data)
    training_ratio=0.65
    training_data=[]
    evaluation_data=[]

    for indice in range(0,total):
        if indice < total *training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data,evaluation_data

def preprocessing_step():
    data= get_all_data()
    processing_data=preprocessing_data(data)
    return split_data(processing_data)

def training_step(data,vectorizer):
    training_text =[data[0] for data in data]
    training_result =[data[1]for data in data]

    training_text= vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text,training_result)

training_data,evaluation_data = preprocessing_step()
vectorizer =CountVectorizer(binary ='true')
classifier =training_step(training_data,vectorizer)

def analyse_text(classifier,vectorizer,text):
    return classifier.predict(vectorizer.transform([text]))


def getNews(data):
    titles=[]
    for x in range(0,15):
        titles.append(data['articles'][x]['title'])
    return titles

def analyse_headlines(data):
    data2=data
    titles = getNews(data2)
    good_headlines=[]
    for x in titles:
        Ans=analyse_text(classifier, vectorizer,x)
        if Ans[0] == '1':
             good_headlines.append(titles.index(x))
    return good_headlines



def main():
        if __name__== "__main__" :
         newsapi = NewsApiClient(api_key='ee558c5dc05c4b6c9604f24d4eeadeb6')
         data = newsapi.get_top_headlines(category='general',
                                          language='en',country='in',
                                          page_size=100)
         print(data['totalResults'])
         good_headlines=analyse_headlines(data)
         for x in good_headlines:
             print (x,"\t",data['articles'][x]['title'],"\n",
                    data['articles'][x]['url'])

main()
