# 2020 BJTU Graduation Project
PARK HOIJAI 18309002
</br>
Beijing Jiaotong University
</br>
Software engineering

## Development Version
* Flask --1.1.1
* SQAlchemy --1.3.1
* Tensorflow --1.15.2
* numpy --1.18.2
* gensim --3.8.1

## Development Tool
* Google Colab Pro https://colab.research.google.com/
* PyCharm 2019.3.2
* Python 3.7
</br>

# Introduction of the project

## 0. Sturcture of the Entire project
I do not share datasets due to copyright.
</br>
![strc](https://github.com/par3k/Graduation/blob/master/img/structure%20of%20the%20entire%20system.png)
</br>
## 1. League Category classification
The dataset had been labeled like this,</br>
![lcc1](https://github.com/par3k/Graduation/blob/master/img/%EC%B6%95%EA%B5%AC%EB%A6%AC%EA%B7%B8%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EB%A5%98.png)
</br>
and here is a Wordcloud based on the datasets from Naver sports news headlines
</br>
![lcc2](https://github.com/par3k/Graduation/blob/master/img/%EC%B6%95%EA%B5%AC%EB%A6%AC%EA%B7%B8%20%EC%9B%8C%EB%93%9C%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C.png)
</br>
There is a map after the word embedding using Word2Vec
</br>
![lcc3](https://github.com/par3k/Graduation/blob/master/img/%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EC%A7%80%EB%8F%84.png)
</br>
## 2. Sentimental Analysis
The dataset had been labeled like this,</br>
![sa_1](https://github.com/par3k/Graduation/blob/master/img/%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EB%A5%98.png)
</br>
and here is a Wordcloud based on the datasets from Naver movie user review comments
</br>
![sa_2](https://github.com/par3k/Graduation/blob/master/img/%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EC%9B%8C%EB%93%9C%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C.png)
</br>
There is a map after the word embedding using Word2Vec
</br>
![sa3](https://github.com/par3k/Graduation/blob/master/img/%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EC%A7%80%EB%8F%84.png)
</br>
## 3. Result of Train (by Bi_LSTM)
There is a accuracy after training (Epochs = 10)
</br> there show pretty higher than i expect. almost over 92% both of them.
</br>
![acc](https://github.com/par3k/Graduation/blob/master/img/accuracy.png)
</br>
## 4. Implement the Web System
1.Main page
</br>
![main1](https://github.com/par3k/Graduation/blob/master/img/1.%20main.png)
</br>
When you click the button, we can see the developer's info.
</br>
![profile](https://github.com/par3k/Graduation/blob/master/img/2.%20profile.png)
</br>
2.Football league category classification function page
</br>
</br>
![main2](https://github.com/par3k/Graduation/blob/master/img/3.%20main2.png)
</br>
When you click the button, we can check the football league category classification function
</br>
</br>
![lc](https://github.com/par3k/Graduation/blob/master/img/4.%20league_classifier.png)
</br>
</br>
You can check the result, when i wrote about Hueng-min Son and Tottenham,
</br>
</br>
they show "EPL" about 98% accuracy.
</br>
</br>
![lc2](https://github.com/par3k/Graduation/blob/master/img/5.%20league_classifier_result.png)
</br>
</br>
3.Comment sentimental analysis function page
</br>
</br>
![main3](https://github.com/par3k/Graduation/blob/master/img/6.%20main3.png)
</br>
When you click the button, we can check the =sentimental analysis function
</br>
</br>
![sa](https://github.com/par3k/Graduation/blob/master/img/7.%20sentimental_analysis.png)
</br>
</br>
You can check the result, when i wrote about Positive one,
</br>
</br>
they show "Positive" about 96% accuracy.
</br>
</br>
![sa2](https://github.com/par3k/Graduation/blob/master/img/8.%20sentimental_analysis_result.png)
</br>
</br>
