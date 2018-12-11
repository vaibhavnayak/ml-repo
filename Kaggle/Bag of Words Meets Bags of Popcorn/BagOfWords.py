# -*- coding: utf-8 -*-
"""
Created on Thu May 10 02:37:06 2018

@author: bkn
"""
import pandas as pd
from bs4 import BeautifulSoup      # Import BeautifulSoup (for removing html tags)
import re                          # For removing regular expressions (punctuation, numbers)
import nltk                        # For removing stopwords
# nltk.download()                    # To download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list

train = pd.read_csv('labeledTrainData.tsv', delimiter = '\t', header=0, quoting=3)

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])

print (train["review"][0])
print (example1.get_text())

# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search

print(letters_only)

lower_case = letters_only.lower()
words = lower_case.split()

print (stopwords.words("english") )

words = [w for w in words if not w in stopwords.words('english')]





