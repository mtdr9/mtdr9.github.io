---
layout: post
title: Kids These Days Really Love Chicken
image: /img/regal_sandwich.jpeg
tags: [poultry, beef, us, consumption]
---


```python
%matplotlib inline

import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams



rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


```


```python
# a note on scraping pitchfork: according to their robots.txt file, scraping is allowed,
# as long as users aren't scraping any of their search pages.
# User-agent: *
# Allow: /
# Disallow: /search/
# Disallow: /search
```

## Scraping Pitchfork for Reviews


```python
#import the artists and albums from 'meddulla's github page
pitchfork_data = pd.read_csv(
    'https://gist.githubusercontent.com/meddulla/8277139/raw/f1595d50cada910d082bc1dfd8ef47ff49910cb3/pitchfork-reviews.csv',
           sep = ',')
pitchfork_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>album</th>
      <th>score</th>
      <th>reviewer</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Burial</td>
      <td>Rival Dealer EP</td>
      <td>9.0</td>
      <td>Larry Fitzmaurice</td>
      <td>http://pitchfork.com/reviews/albums/18820-buri...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Thing</td>
      <td>BOOT!</td>
      <td>7.7</td>
      <td>Aaron Leitko</td>
      <td>http://pitchfork.com/reviews/albums/18735-the-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lumbar</td>
      <td>The First And Last Days Of Unwelcome</td>
      <td>7.7</td>
      <td>Grayson Currin</td>
      <td>http://pitchfork.com/reviews/albums/18705-lumb...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bryce Hackford</td>
      <td>Fair</td>
      <td>7.0</td>
      <td>Nick Neyland</td>
      <td>http://pitchfork.com/reviews/albums/18789-bryc...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waka Flocka Flame</td>
      <td>DuFlocka Rant 2</td>
      <td>7.1</td>
      <td>Miles Raymer</td>
      <td>http://pitchfork.com/reviews/albums/17760-waka...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#input: urls in a pandas series format
#output: review text for each of those links as an array of strings
from bs4 import BeautifulSoup

def get_multiple_reviews(urls):
    if isinstance(urls, pd.Series):
        review_text = []

        for link in urls:
            #pull url html into a beautifulsoup object        
            try:
                r = requests.get(link)
                soup = BeautifulSoup(r.text, 'lxml')
                #pull all paragraph tags/contents with soup.findAll, and then put the text into one string
                review_text.append('\n\n'.join(map(str, [i.text.lower() for i in soup.findAll('p')])))
            except Exception:
                review_text.append(np.nan)
        return review_text
    
    else:
        print('Check inputs - confirm they\'re in pd.Series format')
        return


```


```python
#sample_df = pitchfork_data.sample(5000)
#sample_df['review'] = sample_reviews_merged
#sample_df.to_csv('C:\\Users\\Matt\\Documents\\Python_Scripts\\CS109_Zips\\2013-homework\\HW Practice\\pitchfork_reviews.csv')
```


```python
sample_df = pd.read_csv(
    'C:\\Users\\Matt\\Documents\\Python_Scripts\\CS109_Zips\\2013-homework\\HW Practice\\pitchfork_reviews.csv', 
     sep=',', encoding = 'latin-1', index_col = 'Unnamed: 0')
sample_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>album</th>
      <th>score</th>
      <th>reviewer</th>
      <th>url</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4631</th>
      <td>August Born</td>
      <td>August Born</td>
      <td>7.3</td>
      <td>Matthew Murphy</td>
      <td>http://pitchfork.com/reviews/albums/484-august...</td>
      <td>drag city-issued collaboration between six org...</td>
    </tr>
    <tr>
      <th>4034</th>
      <td>Deerhunter</td>
      <td>Cryptograms</td>
      <td>8.9</td>
      <td>Marc Hogan</td>
      <td>http://pitchfork.com/reviews/albums/9824-crypt...</td>
      <td>best new music\n\nthis atlanta five-piece's sh...</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>Unicycle Loves You</td>
      <td>Unicycle Loves You</td>
      <td>6.5</td>
      <td>Ian Cohen</td>
      <td>http://pitchfork.com/reviews/albums/12057-unic...</td>
      <td>despite making ebullient indie pop, this chica...</td>
    </tr>
    <tr>
      <th>2212</th>
      <td>Flying Lotus</td>
      <td>Cosmogramma</td>
      <td>8.8</td>
      <td>Joe Colly</td>
      <td>http://pitchfork.com/reviews/albums/14198-cosm...</td>
      <td>best new music\n\nthe l.a. producer's head mus...</td>
    </tr>
    <tr>
      <th>4989</th>
      <td>Skalpel</td>
      <td>Skalpel</td>
      <td>7.2</td>
      <td>Jonathan Zwickel</td>
      <td>http://pitchfork.com/reviews/albums/7768-skalpel/</td>
      <td>ninja tune exports skalpel cut up dusty 50s an...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('min/mean/max: ',pitchfork_data.score.min(), pitchfork_data.score.mean(), pitchfork_data.score.max())
print('# reviews: ',len(sample_df))
```

    min/mean/max:  0.0 6.955978497748127 10.0
    # reviews:  5000
    

#### Clean the data
Our review scraper may have imported some NaNs, so let's clean those up


```python
print('NA values: ',sum(sample_df.review.isnull()))
sample_df = sample_df.reset_index(drop=True)
sample_df.drop(sample_df.index[sample_df[sample_df.review.isnull()].index.values], axis=0, inplace=True)
print('NA values: ',sum(sample_df.review.isnull()))

```

    NA values:  2
    NA values:  0
    

## Visualizing the Reviewers


```python
#Let's explore the data we've got a bit
print('# of reviews: ', len(sample_df))
print('# of reviewers: ', len(sample_df.reviewer.unique()))

```

    # of reviews:  4998
    # of reviewers:  234
    


```python
#Histogram to look at the spread of reviewers
plt.title('How Prolific are the Reviewers?')
plt.xlabel('Quantity of Reviews')
plt.ylabel('Number of Reviewers')
plt.hist(sample_df.groupby('reviewer').review.count(), bins=40)
plt.show()
```


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_12_0.png)



```python
top_5_reviewers = sample_df.groupby('reviewer').review.count().nlargest(5)
print('The 5 most prolific reviewers:\n', top_5_reviewers)

```

    The 5 most prolific reviewers:
     reviewer
    Joe Tangari           248
    Stephen M. Deusner    228
    Ian Cohen             176
    Brian Howe            169
    Mark Richardson       149
    Name: review, dtype: int64
    


```python
#pull the names of all reviewers with >n reviews
#use that to get average reviews for relatively prolific reviewers
floor_score = 10
df_count = sample_df.groupby('reviewer').count()
df_mean = sample_df.groupby('reviewer').mean()
plt.title('Average score for reviewers with >%i reviews' %floor_score)
plt.ylabel('# Reviewers')
plt.xlabel('Score')
plt.hist(df_mean[sample_df.groupby('reviewer').count().review>floor_score].score)
plt.show()
```


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_14_0.png)



```python
cutoff_score = 8
print('Percent of reviews that are \'best new music\' (>%i.0): ' %cutoff_score, sum(sample_df.score>cutoff_score)
      /sum(sample_df.score>0)*100)

```

    Percent of reviews that are 'best new music' (>8.0):  16.6099659796
    

## Visualizing the Reviews

First, let's vectorize our dataset and see which words appear frequently


```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sample_df.review)
x = vectorizer.transform(sample_df.review)
x = x.toarray()
len(vectorizer.get_feature_names())
words = pd.DataFrame({'word':vectorizer.get_feature_names(),'freq':x.sum(axis=0)})
print('20 highest frequency words: ', [i for i in words.nlargest(20, 'freq').word.values])

```

    20 highest frequency words:  ['the', 'of', 'and', 'to', 'in', 'that', 'it', 'is', 'on', 'with', 'as', 'for', 'but', 'his', 'like', 'you', 'from', 'this', 'their', 'an']
    


```python
plt.hist(words.freq, bins=200)
plt.title('Nearly every word appears only once')
plt.xlabel('Frequency')
plt.ylabel('Number of words that appear F times')
plt.axis([0,250,0,70000])
plt.show()
```


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_19_0.png)


Stop words aren't very meaningful, and words that appear once in a blue moon are not generalizable, so let's add some limits to the frequency of words

We'll search for the ideal vectorization parameters later; for now let's just try 1e-05 and 0.1.


```python
vectorizer = CountVectorizer(min_df = 1e-05, max_df = 0.1)
vectorizer.fit(sample_df.review)
x = vectorizer.transform(sample_df.review)
x = x.toarray()
len(vectorizer.get_feature_names())
words = pd.DataFrame({'word':vectorizer.get_feature_names(),'freq':x.sum(axis=0)})
print('20 highest frequency words: ', [i for i in words.nlargest(20, 'freq').word.values])

```

    20 highest frequency words:  ['rap', 'jazz', 'blues', 'singles', 'fire', 'disco', 'hits', 'star', 'fi', 'sun', 'york', 'dream', 'eyes', 'electric', '90s', 'machine', 'organ', 'recordings', 'roll', 'riffs']
    


```python
plt.hist(words.freq, bins=400)
plt.title('Stop words eliminated')
plt.xlabel('Frequency')
plt.ylabel('Number of words that appear F times')
plt.axis([0,125,0,45000])
plt.show()
```


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_23_0.png)



```python
print('number of words appearing more than once: ', words[words.freq>1].freq.count())
print('number of words appearing once: ', words[words.freq==1].freq.count())

```

    number of words appearing more than once:  46115
    number of words appearing once:  24334
    

That looks better - no more stop words, and a more reasonable distribution of word frequency.

Note that the graph is zoomed: some words appear more than 120 times.


## Train Classifier


```python
#We'll use this function in our grid search to reconstruct the vectorizer
#critics is the dataframe of reviews, etc.
from sklearn.feature_extraction.text import CountVectorizer

def make_xy(critics, mind=1, maxd=0.99):
    #create y matrix, where y=1 if it's greater than our cutoff score
    bnm = critics.score>cutoff_score
    y = bnm.astype(int).values
    
    #set up the vectorizer, with terms for min_df and max_df
    vectorizer = CountVectorizer(min_df = mind, max_df = maxd)
    vectorizer.fit(critics.review)
    x = vectorizer.transform(critics.review)
    x = x.toarray()
    
    return x, y, vectorizer

X, Y, v = make_xy(sample_df, 1, 0.99)

print('Number of words analyzed: ', len(v.get_feature_names()))
print('Average words per article per the min/max_df above: ', np.array([i.sum() for i in X]).mean())

```

    Number of words analyzed:  71021
    Average words per article per the min/max_df above:  547.547418968
    


```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#Select best parameters (min_df, max_df, alpha)

#cs109 parameter selection. gridsearch above
import sklearn.model_selection
#we'll use the area under the roc curve as a metric
from sklearn.metrics import roc_auc_score

#the grid of parameters to search over
alphas = [.1, 0.5, 1, 3, 5, 10, 50]
min_dfs = [1e-4, 1e-3, 1e-2]
max_dfs = [1e-1, 0.2, 0.5]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
best_max_df = None
max_auc = -np.inf

for a in alphas:
    for min_df in min_dfs:   
        for max_df in max_dfs:
            #configure vectorizer
            #vectorizer = CountVectorizer(min_df = min_df, max_df = max_df)       
            X, Y, v = make_xy(sample_df, min_df, max_df)

            #split up testing/training
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)
            
            #fit the naive bayes model
            nb = MultinomialNB()
            nb.set_params(alpha=a)
            nb.fit(xtrain, ytrain)
            
            #function below implements kfold validation to find best auc
            y_score = nb.predict_proba(xtest)[:,1]
            nll = roc_auc_score(ytest, y_score)
            
            #set best params
            if nll > max_auc:
                max_auc = nll
                best_alpha = a
                best_min_df = min_df
                best_max_df = max_df

```


```python
print(max_auc, '|', best_alpha, '|', best_min_df, '|', best_max_df)

```

    0.770789473684 | 3 | 0.01 | 0.2
    


```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#split our data into training data (80%) and testing (20%)
X, Y, v = make_xy(sample_df, best_min_df, best_max_df)
#provide indices so we can examine later
tts_indices = list(range(len(X)))
xtrain, xtest, ytrain, ytest, indices_train, indices_test = train_test_split(
    X, Y, tts_indices, test_size = 0.2, random_state = 10)

#fit the naive bayes model
nb = MultinomialNB(alpha=best_alpha)
nb.fit(xtrain, ytrain)

#test training prediction
print('training accuracy: ', sum(nb.predict(xtrain)==ytrain)/len(ytrain))
print('testing accuracy: ', sum(nb.predict(xtest)==ytest)/len(ytest))
#well this model is most definitely overfit - training performance is much better than testing
#also with just 15% of music as bnm, I'm concerned with 81% accuracy

```

    training accuracy:  0.907453726863
    testing accuracy:  0.816
    

#### Testing/validation


```python
#ROC score!
from sklearn.metrics import roc_auc_score
y_score = nb.predict_proba(xtest)[:,1]
roc_auc_score(ytest, y_score)

```




    0.75404794825879673




```python
#ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresh = roc_curve(ytest, y_score)
plt.plot(fpr, tpr)
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))

```




    [<matplotlib.lines.Line2D at 0x13cb3d967f0>]




![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_33_1.png)



```python
#calculate youden's j and identify the threshold at which it's maximized
youden_j = tpr + (1-fpr) -1
ideal_thresh = thresh[np.where(youden_j == youden_j.max())]
plt.plot(np.linspace(1, len(thresh),len(thresh)),thresh)
plt.plot(np.linspace(1, len(thresh), len(thresh)),youden_j)
plt.plot(np.where(youden_j == youden_j.max()),ideal_thresh,'o', c='r')
print('ideal threshold: ',ideal_thresh)
plt.show()
```

    ideal threshold:  [ 0.01358134]
    


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_34_1.png)



```python
#let's assess whether our model is well-calibrated (i.e. whether its confidence % is about equal to its accuracy %)
hst = plt.hist(nb.predict_proba(xtest)[:,1], bins=20)

```


![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_35_0.png)



```python
"""As demo'd above, Naive Bayes tends to push probabilties to 0 or 1, mainly because it makes the assumption 
 that features are conditionally independent, which is not true here."""
```




    "As demo'd above, Naive Bayes tends to push probabilties to 0 or 1, mainly because it makes the assumption \n that features are conditionally independent, which is not true here."




```python
bin_counts = hst[0]
probabilities = nb.predict_proba(xtest)
digits = np.digitize(probabilities[:,1], hst[1])

#next step is to figure out what % of each bin in our test data is fresh
#so we'll need to assign ytest to bins per digits
binframe = pd.DataFrame({'y':ytest, 'bin':digits})
bnm_prob = binframe.groupby('bin').sum()/binframe.groupby('bin').count()

#number of reviews in each bin divided by the total number of (test) reviews
pct_df = binframe.groupby('bin').count()/binframe.bin.count()

#pct_df won't have the right # of items if the bins aren't nicely distributed, so we'll pad it
ind = list(range(1,len(hst[0])+2))
indy_df = pd.DataFrame({'b':np.zeros(len(ind))}, index=ind)
indy_df = indy_df.join(pct_df)
indy_df = indy_df.max(axis=1)

#plot the cumulative distribution function
plt.xlabel('P(fresh)')
plt.ylabel('P(predicted fresh)')
plt.title('The model is overconfident at both extremes, typical of Naive Bayes models')

#plt.plot(hst[1][:-1], np.cumsum(pct_df.values),'.')
plt.plot(hst[1], np.cumsum(indy_df.values),'.')
plt.plot(hst[1][:-1],hst[1][:-1], alpha=.5)
plt.legend(['Distribution of predicted values', 'True distribution'])

```




    <matplotlib.legend.Legend at 0x13cb3b2e748>




![png](HW3_practice_pitchfork_files/HW3_practice_pitchfork_37_1.png)


#### Looking at reviews the model got wrong


```python
#create list with the indices and test results (w)
all_pred = list(zip(indices_test,nb.predict(xtest)==ytest))
#Where false, the prediction didn't agree with reality
#[i for i in correct_pred[1]==False]
incorrect_pred = [i[0] for i in correct_pred if i[1]==False]
print(sample_df.iloc[incorrect_pred[0],:],'\n\n', 
      sample_df.iloc[incorrect_pred[1],:],'\n\n', sample_df.iloc[incorrect_pred[2],:])
```

    artist                                           Blake Miller
    album                                      Together With Cats
    score                                                     7.4
    reviewer                                           Brian Howe
    url         http://pitchfork.com/reviews/albums/9769-toget...
    review      with his murky arrangements, overlapping false...
    Name: 3205, dtype: object 
    
     artist                                       El Perro Del Mar
    album                                        El Perro del Mar
    score                                                     8.1
    reviewer                                   Stephen M. Deusner
    url         http://pitchfork.com/reviews/albums/9087-el-pe...
    review      on her second album, swedish singer-songwriter...
    Name: 1562, dtype: object 
    
     artist                                            Whiskeytown
    album                                               Pneumonia
    score                                                     8.1
    reviewer                                         Ryan Kearney
    url         http://pitchfork.com/reviews/albums/8645-pneum...
    review      categorization has been widely accepted as a p...
    Name: 4177, dtype: object
    


```python
sample_df.iloc[incorrect_pred,2].mean()
#average incorrect review score of 7.64, which is fairly close to the dividing line of 8!
```




    7.643478260869569



## Some interpretation


```python
#testing to see which words indicated good results vs. poor
words = np.array(v.get_feature_names())
x = np.eye(xtest.shape[1])
probs = nb.predict_log_proba(x)[:, 0]
ind = np.argsort(probs)
```


```python
print('good words: ',words[ind[:10]])
print('bad words: ',words[ind[-10:]])
```

    good words:  ['reissue' 'remastered' 'pavement' 'rev' 'staggering' 'reissued' 'columbia'
     'reissues' '1969' 'blake']
    bad words:  ['competent' 'affectations' 'annoying' 'cure' 'problem' 'frustrating'
     'forgettable' 'tack' 'fails' 'decent']
    

Pitchfork really likes reissues! They are also not impressed with anything that's merely 'decent' or 'competent'.


```python
#custom review testing
print(nb.predict_proba(v.transform(
    ['awful tinny repetitive boring dull'])))
print(nb.predict_proba(v.transform(
    ['rich experimental complex excellent luxurious reissue from columbia'])))
print(nb.predict_proba(v.transform(
    ['kanye'])))

```

    [[ 0.94119048  0.05880952]]
    [[ 0.12944599  0.87055401]]
    [[ 0.70460993  0.29539007]]
    

## Various testing, obsolete sections
I was able to get it to work outside of the gridsearch but not within



```python
cv_score(nb, x, y)
```




    array([-0.68242805, -0.67461008, -0.67176271])




```python
nb = MultinomialNB()
nb.set_params(alpha=.1)
nb.fit(xtrain, ytrain)
nb.predict(xtrain)
```




    array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
           1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
           0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
           1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])




```python
vectorizer = CountVectorizer(min_df = min_df)       
X, Y, v = make_xy(sample_df)
        
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)
#fit the naive bayes model
nb = MultinomialNB(alpha=1)
#nb.set_params(alpha=1)
nb.fit(xtrain, ytrain)
#nll = cv_score(nb, X, Y) #function below already implements kfold validation
nll = sklearn.cross_validation.cross_val_score(nb, xtest, ytest, scoring='neg_log_loss')
sum(nll)
```




    -38.903566818443352




```python
alphas = [0, .1, 1, 5, 10, 50]
for a in alphas:
    print(type(a))
```

    <class 'int'>
    <class 'float'>
    <class 'int'>
    <class 'int'>
    <class 'int'>
    <class 'int'>
    


```python
import sklearn.model_selection
ll = sklearn.cross_validation.cross_val_score(nb, xtest, ytest, scoring='neg_log_loss')
sum(ll)
#-12... pretty bad! 0.683 is as good as random guessing...
#pre optimization: array([-12.71206784,  -9.92044569,  -9.87049272])
#the sum of neg log loss was -37.09
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-223-3cbff8e1365e> in <module>()
          1 import sklearn.model_selection
    ----> 2 ll = sklearn.cross_validation.cross_val_score(nb, xtest, ytest, scoring='neg_log_loss')
          3 sum(ll)
          4 #-12... pretty bad! 0.683 is as good as random guessing...
          5 #pre optimization: array([-12.71206784,  -9.92044569,  -9.87049272])
    

    AttributeError: module 'sklearn' has no attribute 'cross_validation'



```python
#cs109 parameter selection. gridsearch above
import sklearn.model_selection
#we'll use the area under the roc curve as a metric
from sklearn.metrics import roc_auc_score

#the grid of parameters to search over
alphas = [.1, 0.5, 1, 3, 5, 10, 50]
min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
max_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
best_max_df = None
max_auc = -np.inf

for a in alphas:
    for min_df in min_dfs:   
        for max_df in max_dfs:
            vectorizer = CountVectorizer(min_df = min_df, max_df = max_df)       
            X, Y, v = make_xy(sample_df)

            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)
            
            #fit the naive bayes model
            nb = MultinomialNB()
            nb.set_params(alpha=a)
            nb.fit(xtrain, ytrain)
            
            #function below implements kfold validation to find best auc
            y_score = nb.predict_proba(xtest)[:,1]
            nll = roc_auc_score(ytest, y_score)
            #nll = sklearn.cross_validation.cross_val_score(nb, xtest, ytest, scoring='neg_log_loss')
            if nll > max_auc:
                max_auc = nll
                best_alpha = a
                best_min_df = min_df
                best_max_df = max_df

```


```python
#this cell obsolete

from sklearn.cross_validation import KFold

def cv_score(clf, x, y, score_func=0):
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold):
        nb.fit(x[train], y[train])
        result += sklearn.cross_validation.cross_val_score(
            clf, x[test], y[test], scoring='neg_log_loss')
    return result/nfold


```

# To Do
Add max_df as a term in our vectorizer to our gridsearch
   - The model's performance improves at around 0.3
   - Although that's without cross-fold validation



