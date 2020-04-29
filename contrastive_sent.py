'''
ENTITY SENTIMENT
This file contains various functions that, given a text and word within the text, estimates sentiment towards the particular word.
Typically the text should be a sentence and the word should be an entity of interest, for example a person, place, or thing.
Determination of sentiment is implemented in 3 different ways:
1. Split method: sentence is split on commas and comparison words. Sentiment is determined from split that contains the word of interest.
2. Neighborhood method: finds the word in the sentence and looks at nearby words to determine sentiment.
3. Tree method: creates a dependency tree of the sentence using the Stanford dependency parser. Sentiment is determined by looking at
words that appear nearby in the tree to the word of interest.
The function compile_ensemble_sentiment combines all three of these methods and is generally the most robust way to determine sentiment
towards an entity.
'''


# import spacy
# from spacy import displacy
# import en_core_web_sm
# nlp = en_core_web_sm.load()

# from model import load_n_predict
# from aen import AEN_BERT


# model = load_n_predict(AEN_BERT)

def search(a_str, sub):
    '''
    Find all matches of sub within a_str. Returns starting index of matches.
    '''
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def split_sentiment(sentence, entity):
    '''
    Split the sentence by comparison words and commas. Determine which sections the entity is in.
    Return average sentiment for those sections.
    '''
    # List of comparison words
    contrastive_words = ['but', 'however', 'albeit', 'although', 'in contrast', 'in spite of', 'though', 'on one hand', 'on the other hand',
                  'then again', 'even so', 'unlike', 'which', 'while', 'conversely', 'nevertheless', 'nonetheless', 'notwithstanding', 'yet']

    # Lowercase sentence and split on commas
    sentence = sentence.lower()
    sentence = sentence.split(',')

    # Iterate through sections and split them based on contrastive words
    splits = []
    for section in sentence:

        all_comps = []
        for word in contrastive_words:
            # Use find all function to find location of contrastive words
            all_comps += list(search(section, word))

        # Sort list of contrastive words indexes
        all_comps.sort()

        # Split the section and append to splits
        last_split = 0
        for comp in all_comps:
            splits.append(section[last_split:comp])
            last_split = comp
        splits.append(section[last_split:])

    # Find the sections where the entity has been named
    # Add sentiment for that section to list
    sentiment = []
    for section in splits:
        if entity.lower() in section:
            # remove entity from section
            cleaned_section = section.replace(entity.lower(), '')
            # return cleaned_section
            sentiment.append(model.predict(cleaned_section, entity, cleaned_section))
    if int(str(sentiment).strip('[]')) == 0:
      print('Sentiment for {} is Negative'.format(entity)) 
    else:
      print('Sentiment for {} is Positive'.format(entity))           



# def display_dep_tree(sentence, opt = True):
#     doc = nlp(sentence)
#     options = {"compact": True, "bg": "#09a3d5",
#            "color": "white", "font": "Source Sans Pro"}
#     if opt == True:
#         displacy.serve(doc, style="dep", options=options)
#     else:
#         displacy.serve(doc, style="dep")


# def dep_tree_sentiment(sentence):
#     doc = nlp(sentence)

#     for token in doc:
#         print(token.text, token.pos_, token.dep_)


# def compile_ensemble_sentiment(sentence, entity, tree_dict = False):
#     '''
#     Determines sentiment using three different methods: neighborhood, tree, and split.
#     Sentiment for each method is compiled to return a final sentiment.
#     '''
#     compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}


#     tree_polarity = dep_tree_sentiment(sentence, entity, tree_dict = tree_dict)
#     split_polarity = split_sentiment(sentence, entity)

#     all_sentiments = [tree_polarity, split_polarity]

#     # Add up all sentiment
#     for sent in all_sentiments:
#         for key in compiled_sentiment.keys():
#             compiled_sentiment[key] += sent[key]

#     # Divide by 3 to get average
#     for key in compiled_sentiment.keys():
#         compiled_sentiment[key] = compiled_sentiment[key] / 2

#     return compiled_sentiment, tree_dict
