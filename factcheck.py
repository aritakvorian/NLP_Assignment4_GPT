# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import nltk
import gc
from rouge_score import rouge_scorer


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        entailment_prob = probabilities[0]
        neutral_prob = probabilities[1]
        contra_prob = probabilities[2]

        if entailment_prob > max(neutral_prob, contra_prob):
            prediction = "S"
        else:
            prediction = "NS"

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something
        return prediction

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def __init__(self, classification_threshold=0.025, nlp=None, stop_words=None):
        self.classification_threshold = classification_threshold
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()

        if stop_words is None:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        else:
            self.stop_words = stop_words

        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

    def preprocessing(self, text):
        doc = self.nlp(text.lower())
        tokens = []

        for token in doc:
            if token.lemma_ in self.stop_words:
                continue
            if token.is_alpha is False:
                continue

            stemmed_token = self.stemmer.stem(token.lemma_)
            tokens.append(stemmed_token)

            #tokens.append(token.lemma_)

        # Bigrams
        #tokens_with_bigrams = tokens + ["_".join(bigram) for bigram in nltk.bigrams(tokens)]
        tokens_with_bigrams = tokens

        return set(tokens_with_bigrams)

    def jaccard_similarity(self, a, b):
        overlap = a.intersection(b)
        union = a.union(b)
        if union:
            return len(overlap)/len(union)
        else:
            return 0

    def harmonic_jaccard(self, a, b):
        overlap = a.intersection(b)
        union = a.union(b)

        # Harmonic mean of precision, recall, and jaccard
        jaccard = len(overlap)/len(union)
        precision = len(overlap) / len(b)
        recall = len(overlap) / len(b)

        if jaccard + precision + recall == 0:
            return 0
        else:
            similarity = 3 * (precision * recall * jaccard) / (precision + recall + jaccard)
            return similarity

    def sentence_level_similarity(self, fact_tokens, passage):
        sentences = nltk.sent_tokenize(passage)
        max_similarity = 0
        for sentence in sentences:
            sentence_tokens = set(self.preprocessing(sentence))

            # If fact fully in sentence
            if fact_tokens.issubset(sentence_tokens):
                return 1

            similarity = self.jaccard_similarity(fact_tokens, sentence_tokens)
            if similarity > max_similarity:
                max_similarity = similarity

        return max_similarity

    def predict(self, fact: str, passages: List[dict]) -> str:
        input_tokens = set(self.preprocessing(fact))
        max_similarity = 0

        for passage in passages:
            text_tokens = set(self.preprocessing(passage["text"]))
            similarity = self.jaccard_similarity(input_tokens, text_tokens)
            #similarity = self.sentence_level_similarity(input_tokens, passage["text"])
            if similarity > max_similarity:
                max_similarity = similarity

        if max_similarity >= self.classification_threshold:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:

        for passage in passages:
            sentences = passage["text"].split(". ")

            for sentence in sentences:
                if sentence:
                    result = self.ent_model.check_entailment(sentence, fact)
                    if result == "S":
                        return result

        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

