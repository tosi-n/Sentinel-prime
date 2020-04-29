Sentinel-prime
================
Tosin Dairo
APR 29, 2020

Attention encoder network for entity-based sentiment analysis

<hr>

</hr>

#### Background

The goal of the exercise is to design a state-of-the-art model to solve
the problem of entity-based sentiment analysis. Entity-based sentiment
analysis is widely used to automatically detect the opinion expressed
towards named entities. It is particularly social and news media
analytics

#### Problem definition

Given a sentence and a named entity contained within a sentence, predict
the sentiment towards the named entity

#### Targeted Goals

  - Handling unseen vocabulary
  - Predict the sentiment of an entity
  - Predict contrastive sentences
  - K-fold cross-validation
  - Performance statistics

#### Technical Solution

With constant improvement and development in Natural Language Processing
seen with deep learning application. Tranformer’s encoder decoder design
have adopted attention mechanisms, which have enforced the model to pay
more attention to input sentences with relations to their target entity.
Based on the targeted goal to handle unseen vocabulary, text is divided
into a limited set of common sub-word units known wordpieces for both
input and output. A data-driven approach that is guarantees to generate
a deterministic segmentation for any possible sequence of characters is
adopted using a wordpiece model (WPM) like
[BERT](https://arxiv.org/abs/1810.04805).

In order to achieve an entity-based sentiment analysis, an attentional
encoder network
[AEN](https://www.researchgate.net/publication/331343006_Attentional_Encoder_Network_for_Targeted_Sentiment_Classification)
is put together to calculate the hidden states and realtionship between
entity and input sentences. AEN model enhances the performance of basic
BERT model and obtains new state-of-the-art results. This attention
based technique outperforms traditional machine learning methods and
rule-based methods which are usually heavily reliant on feature
engineering works which is not so productive in practice.

Attentional encoder network (AEN), in this implementation is made up of
BERT embedding layer, an attentional encoder layer, a target-specific
attention layer, and an output layer.

Trained model on GPU to attain the below metrics measuring accuracy,
precision, recall and F1-score.

![Performance
statistics](/Users/tosi-n/Downloads/metrics_evaluation.png)

A k-fold cross-validation is used estimate how the model is expected to
perform in general when used to make predictions on data not used during
the training of the model. This helps to achieve less biased or less
optimistic estimate of the trained model compared to simple train/test
split.

Predicting contrastive sentences whenever 2 sentiments and entities come
up within a sentence is solved by compiling a list of contrastive words
used in senteences, then search through input sentence for contrastive
words and split then sentence into 2 at that point. This then allows the
AEN model to concetrate on the part of the sentence entity appears and
search for the sentiment to determine it’s polarity.

#### Limitations

  - Require more data to pretrain model for improved F1-score as
    hyperparameters depends on data volume
  - Split technique for contrastive sentences is limited to sentiment
    existing in the split half

> Original adaptation and code reference can be found
> [HERE](https://github.com/songyouwei/ABSA-PyTorch)

#### What Next

#### References

  - BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina
    Toutanova
  - Attentional Encoder Network for Targeted Sentiment Classification -
    Youwei Song, Jiahai Wang, Tao Jiang, Zhiyue Liu, Yanghui Rao
  - <https://github.com/songyouwei/ABSA-PyTorch>
  - <http://jalammar.github.io/illustrated-transformer/>
