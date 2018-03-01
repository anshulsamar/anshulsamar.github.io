---
layout: narrative
title: End to End Speech Recognition
author: Anshul Samar
date: 2014-08-28
mydate: Aug 2014
---

The summer after my sophomore year, I was introduced to deep learning for the first time and
got to train end to end RNNs for speech recognition. This was part of
my CURIS project in the Andrew Ng Deep Learning Lab. 

Speech recogniton is an important subfeld of machine learning with
vast applicability to human computer interacton, telephone and audio
transcripton, deaf assistance, and government intelligence – popular examples
include Siri, Nuance, and Google Now. However, due to variance in
speaking styles, transcribing human-human conversatons is
challenging. Traditonal algorithms (Hidden Markov models, Gaussian
mixture models/RNNs) require an extra step of determining which window
of speech corresponds to which English word. However, these alignments
are not readily available leading to a complicated error prone cycle
of alignment generaton and training. In Connectonist Temporal
Classifcaton (CTC) [2], a recurrent neural network is trained to
output characters directly from speech features, without relying on
any additonal inputs or alignments, making it invariant to speaker-to-speaker
diferences. By training on large networks (deep learning) and many hours of
speech, we allow our models to learn the nuances of speech without having to
provide features from domain specifc knowledge and be wholly data driven.

Thanks to Awni Hannun, Andrew Maas, Ziang Xie, Peng Qiu, and Andrew Ng
for their mentorship and help. Over the summer, I:

1. Trained RNNs on Wall Street Journal 
Corpus (~81 hours, news reading) and Switchboard Corpus (~300
hours, telephone conversatons)

2. Implemented dropout and investigated effect of different feature sets (mel flter bank cepstral coefcients, filter
banks, context windows for past/future informaton) on Wall Street Journal
performance

3. Built a confgurable feature pipeline to easily create many different
training sets and worked on an internal neural network experiment manager to
easily conduct many deep learning experiments on cluster computers.

![Simba]({{ site.baseurl }}{{ "/assets/speech_examples.png"  }}){: .center-image}

Poster <a href="https://web.stanford.edu/~asamar/asamar2014.pdf">
here.</a>

<a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.6306&rep=rep1&type=pdf">[1] </a>Graves, A., Fernandez, S., Gomez, F., and Schmidhuber,
J. Connectonist temporal classifcaton:
Labeling unsegmented sequence data with recurrent neural networks. In
ICML, pp. 369–376. ACM, 2006.




