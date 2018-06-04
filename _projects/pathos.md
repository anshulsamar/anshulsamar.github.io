---
layout: narrative
title: Sentiment Analysis on Headlines
author: Anshul Samar
date: 2013-12-13
mydate: Dec 2013
---

Determining emotion in a piece of text is a subset of sentiment
analysis, a field which has largely
focused on binary classification of text as positive or
negative. Classifying sentences in one of six major emotion categories
(anger, disgust, fear, joy, sadness, and surprise), however, can
offer more insight and contribute to political polling, marketing, and stock
market predictions.

In this study, we focus on the first of these six
emotions, anger, and develop a model to predict whether or not a
reader will feel anger after reading a headline. Our model involves
NLP tools - stemming, named entity recognition, and part of speech
tagging - and a bag of words model run on a SVM with a linear kernel
and a Naive Bayes Classifier with Laplace smoothing. We developed
experimental models to test on smaller subsets of data including: PCA,
to reduce dimensionality of data, a synonym algorithm, to expand each
headline allowing our model to classify with respect to more words,
and a co-occurrence model, to score each headline based on
co-occurrences of words with the word `anger' in the New York Times
archives.

We also built an emotion headline generation model with
Markov chains to see if we can automatically create angry
headlines. With these models, we hope to gain greater insight into
methods that one can use to detect emotion within text.

This project was developed with Bharad Raghavan. See project <a
href="https://github.com/anshulsamar/Pathos">here.</a>

