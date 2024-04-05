import json
from collections import Counter
from pprint import pprint
from operator import indexOf
import re
from typing import Optional
import numpy as np
import nltk
from nltk.data import find
from copy import deepcopy
import gensim
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
nltk.download("word2vec_sample")

########-------------- PART 1: LANGUAGE MODELING --------------########


END_TOKEN = "<EOS>"


class WordCandidate:

    def __init__(self, prev: "Sentence", word: str, prob: float) -> None:
        self.prev = prev
        self.word = word
        self.prob = prob

    def __repr__(self) -> str:
        return f"word: {self.word}, prob: {self.prob}"


class Sentence:

    def __init__(self, prefix=""):
        self.value: str = prefix

    def __repr__(self) -> str:
        return self.value

    def __len__(self) -> int:
        return len(self.value.split())

    def get_last_bigram(self) -> tuple:
        return tuple(self.value.split()[-2:])

    def add(self, word) -> "Sentence":
        self.value += " " + word
        return self

    def mark_as_done(self):
        self.value += " " + END_TOKEN

    def is_done(self):
        return self.value.split()[-1] == END_TOKEN

    def clone(self):
        return deepcopy(self)


class NgramLM:

    def __init__(self):
        """
        N-gram Language Model
        """
        # Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram = {}

        # Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram_weights = {}

    def load_trigrams(self):
        """
        Loads the trigrams from the data file and fills the dictionaries defined above.

        Parameters
        ----------

        Returns
        -------
        """
        with open("data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
            lines = f.readlines()
            for line in lines:
                word1, word2, word3, count = line.strip().split()
                if (word1, word2) not in self.bigram_prefix_to_trigram:
                    self.bigram_prefix_to_trigram[(word1, word2)] = []
                    self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
                self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
                self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

    def top_next_word(self, word1, word2, n=10):
        """
        Retrieve top n next words and their probabilities given a bigram prefix.

        Parameters
        ----------
        word1: str
                The first word in the bigram.
        word2: str
                The second word in the bigram.
        n: int
                Number of words to return.

        Returns
        -------
        next_words: list
                The retrieved top n next words.
        probs: list
                The probabilities corresponding to the retrieved words.
        """

        next_words = self.bigram_prefix_to_trigram[(word1, word2)]

        counts = self.bigram_prefix_to_trigram_weights[(word1, word2)]
        total_count = sum(counts)
        probs = [c / total_count for c in counts]

        sorted_words = sorted(
            list(zip(next_words, probs)), key=lambda t: t[1], reverse=True
        )[:n]

        next_words = [word for word, _ in sorted_words]
        probs = [count for _, count in sorted_words]

        return next_words, probs

    def sample_next_word(self, word1, word2, n=10):
        """
        Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

        Parameters
        ----------
        word1: str
                The first word in the bigram.
        word2: str
                The second word in the bigram.
        n: int
                Number of words to return.

        Returns
        -------
        next_words: list
                The sampled n next words.
        probs: list
                The probabilities corresponding to the retrieved words.
        """

        words = self.bigram_prefix_to_trigram[(word1, word2)]
        counts = self.bigram_prefix_to_trigram_weights[(word1, word2)]
        total_count = sum(counts)

        all_probs = [c / total_count for c in counts]

        next_words = [np.random.choice(words, p=all_probs) for _ in range(n)]
        probs = [all_probs[indexOf(words, word)] for word in next_words]

        return next_words, probs

    def _get_sentence_prob(self, sentence: str, start_at: int = 2, max_length = None) -> float:
        # Split the sentence into words

        words = sentence.split()

        # Assume the sentence already includes start and end tokens
        total_probability = 1.0

        # Iterate through the sentence to compute the probability of each trigram
        for i in range(start_at, len(words)):
            if max_length is not None and i == max_length:
                continue

            bigram_prefix = (words[i - 2], words[i - 1])
            next_word = words[i]

            # Retrieve the list of possible next words and their weights for the bigram prefix
            next_words = self.bigram_prefix_to_trigram.get(bigram_prefix, [])
            weights = self.bigram_prefix_to_trigram_weights.get(bigram_prefix, [])

            # Find the index of the next word in the list of possible next words
            total_bigram_count = sum(weights)
            if next_word in next_words:
                next_word_index = next_words.index(next_word)
                next_word_count = weights[next_word_index]

                # Calculate the probability of the next word given the bigram prefix
                word_probability = next_word_count / total_bigram_count
            else:
                # If the next word does not follow the bigram in the training data, don't account for it
                word_probability = 1

            # Multiply the total probability by the probability of the current trigram
            total_probability *= word_probability

        return total_probability 

    def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
        """
        Generate sentences using beam search.

        Parameters
        ----------
        prefix: str
                String containing two (or more) words separated by spaces.
        beam: int
                The beam size.
        sampler: Callable
                The function used to sample next word.
        max_len: int
                Maximum length of sentence (as measure by number of words) to generate (excluding "<EOS>").

        Returns
        -------
        sentences: list
                The top generated sentences
        probs: list
                The probabilities corresponding to the generated sentences
        """

        prefix_len = len(prefix.split())
        sentences = []
        curr_sentences = [Sentence(prefix)]

        while any([not s.is_done() for s in curr_sentences]):
            candidates: list[WordCandidate] = []

            for i in range(len(curr_sentences)):
                curr = curr_sentences[i]

                if curr.is_done():
                    sentences.append(curr)
                    continue

                if not curr.is_done() and len(curr) == max_len:
                    curr.mark_as_done()
                    continue

                word1, word2 = curr.get_last_bigram()
                next_words, next_probs = sampler(word1, word2, n=beam)
                candidates += [
                    WordCandidate(
                        curr,
                        word,
                        prob * self._get_sentence_prob(str(curr), start_at=prefix_len),
                    )
                    for word, prob in zip(next_words, next_probs)
                ]

            if not candidates:
                break

            num_candidates_need = len(
                [s for s in curr_sentences if not s.is_done()]
            ) if len(curr_sentences) != 1 else beam

            top_candidates = sorted(candidates, key=lambda c: c.prob, reverse=True)[
                :num_candidates_need
            ]

            curr_sentences = [s for s in curr_sentences if s.is_done()] + [
                candidate.prev.clone().add(candidate.word)
                for candidate in top_candidates
            ]

        sentences = [str(s) for s in curr_sentences]
        probs = [
            self._get_sentence_prob(str(s), start_at=prefix_len, max_length=max_len) for s in curr_sentences
        ]

        zipped = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)
        sentences, probs = [[i for i, _ in zipped], [j for _, j in zipped]]

        return sentences, probs


#####------------- CODE TO TEST YOUR FUNCTIONS FOR PART 1

# Define your language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()

# print("------------- Evaluating top next word prediction -------------")
# next_words, probs = language_model.top_next_word("middle", "of", 10)
# for word, prob in zip(next_words, probs):
#     print(word, prob)
# # Your first 5 lines of output should be exactly:
# # a 0.807981220657277
# # the 0.06948356807511737
# # pandemic 0.023943661971830985
# # this 0.016901408450704224
# # an 0.0107981220657277
#
# print("------------- Evaluating sample next word prediction -------------")
# next_words, probs = language_model.sample_next_word("middle", "of", 10)
# for word, prob in zip(next_words, probs):
#     print(word, prob)
# # My first 5 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # a 0.807981220657277
# # pandemic 0.023943661971830985
# # august 0.0018779342723004694
# # stage 0.0018779342723004694
# # an 0.0107981220657277
#
print("------------- Evaluating beam search -------------")
sentences, probs = language_model.generate_sentences(
    prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word
)
for sent, prob in zip(sentences, probs):
    print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

sentences, probs = language_model.generate_sentences(
    prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word
)
for sent, prob in zip(sentences, probs):
    print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> biden calls for a 30 bonus URL #cashgem #cashappfriday #stayathome <EOS> 0.0002495268686322749
# <BOS1> <BOS2> biden says all u.s. governors should mandate masks <EOS> 1.6894510541025754e-05
# <BOS1> <BOS2> biden says all u.s. governors question cost of a pandemic <EOS> 8.777606198953028e-07

sentences, probs = language_model.generate_sentences(
    prefix="<BOS1> <BOS2> wear a mask", beam=10, sampler=language_model.top_next_word
)
for sent, prob in zip(sentences, probs):
    print(sent, prob)
print("#########################\n")
# sentences, probs = language_model.generate_sentences(
#     prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word
# )
# for sent, prob in zip(sentences, probs):
#     print(sent, prob)
# print("#########################\n")
# # My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# # <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# # <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05
#
# sentences, probs = language_model.generate_sentences(
#     prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word
# )
# for sent, prob in zip(sentences, probs):
#     print(sent, prob)
# # My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # <BOS1> <BOS2> biden is elected <EOS> 0.001236227651321991
# # <BOS1> <BOS2> biden dropping ten points given trump a confidence trickster URL <EOS> 5.1049579351466146e-05
# # <BOS1> <BOS2> biden dropping ten points given trump four years <EOS> 4.367575122292103e-05


########-------------- PART 2: Semantic Parsing --------------########


class SemanticParser:
    def __init__(self):
        """
        Basic Semantic Parser
        """
        self.parser_files = "data/semantic-parser"
        self.train_data = []
        self.test_questions = []
        self.test_answers = []
        self.intents = set()
        self.word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            self.word2vec_sample, binary=False
        )
        self.classifier = LogisticRegression(random_state=42, multi_class="multinomial")

        # Let's stick to one target intent.
        self.target_intent = "AddToPlaylist"
        self.target_intent_slot_names = set()
        self.target_intent_questions = []

    def load_data(self):
        """
        Load the data from file.

        Parameters
        ----------

        Returns
        -------
        """
        with open(f"{self.parser_files}/train_questions_answers.txt") as f:
            lines = f.readlines()
            for line in lines:
                self.train_data.append(json.loads(line))

        with open(f"{self.parser_files}/val_questions.txt") as f:
            lines = f.readlines()
            for line in lines:
                self.test_questions.append(json.loads(line))

        with open(f"{self.parser_files}/val_answers.txt") as f:
            lines = f.readlines()
            for line in lines:
                self.test_answers.append(json.loads(line))

        for example in self.train_data:
            self.intents.add(example["intent"])

    def predict_intent_using_keywords(self, question):
        """
        Predicts the intent of the question using custom-defined keywords.

        Parameters
        ----------
        question: str
                The question whose intent is to be predicted.

        Returns
        -------
        intent: str
                The predicted intent.
        """
        keywords = {
            "GetWeather": [
                "weather",
                "degree",
                "hot",
                "warm",
                "cold",
                "sunny",
                "cloudy",
                "sun",
                "get",
                "what",
                "temperature",
                "forecast",
                "rain",
                "snow",
                "wind",
                "celcius",
                "farenheit",
                "like",
                "sentencetoday",
            ],
            "BookRestaurant": [
                "restaurant",
                "book",
                "eat",
                "dinner",
                "lunch",
                "food",
                "breakfast",
            ],
            "AddToPlaylist": ["add", "music", "playlist", "song"],
        }

        counter = {k: 0 for k in keywords}
        for k, words in keywords.items():
            for word in question.split():
                if word.lower() in words:
                    counter[k] += 1

        intent = ""
        max_count = 0
        for k, v in counter.items():
            if v >= max_count:
                intent = k
                max_count = v

        return None if max_count == 0 else intent

    def evaluate_intent_accuracy(self, prediction_function_name):
        """
        Gives intent wise accuracy of your model.

        Parameters
        ----------
        prediction_function_name: Callable
                The function used for predicting intents.

        Returns
        -------
        accs: dict
                The accuracies of predicting each intent.
        """
        correct = Counter()
        total = Counter()
        for i in range(len(self.test_questions)):
            q = self.test_questions[i]
            gold_intent = self.test_answers[i]["intent"]
            if prediction_function_name(q) == gold_intent:
                correct[gold_intent] += 1
            total[gold_intent] += 1
        accs = {}
        for intent in self.intents:
            accs[intent] = (correct[intent] / total[intent]) * 100
        return accs

    def get_sentence_representation(self, sentence):
        """
        Gives the average word2vec representation of a sentence.

        Parameters
        ----------
        sentence: str
                The sentence whose representation is to be returned.

        Returns
        -------
        sentence_vector: np.ndarray
                The representation of the sentence.
        """
        all = [
            self.word2vec_model[word]
            for word in sentence.split()
            if word in self.word2vec_model
        ]
        sentence_vector = np.mean(all, axis=0)

        return sentence_vector

    def train_logistic_regression_intent_classifier(self):
        """
        Trains the logistic regression classifier.

        Parameters
        ----------

        Returns
        -------
        """

        x_train = [
            self.get_sentence_representation(line["question"])
            for line in self.train_data
        ]
        y_train = np.array([line["intent"] for line in self.train_data])

        self.classifier.fit(x_train, y_train)

    def predict_intent_using_logistic_regression(self, question):
        """
        Predicts the intent of the question using the logistic regression classifier.

        Parameters
        ----------
        question: str
                The question whose intent is to be predicted.

        Returns
        -------
        intent: str
                The predicted intent.
        """
        return self.classifier.predict([self.get_sentence_representation(question)])[0]

    def get_target_intent_slots(self):
        """
        Get the slots for the target intent.

        Parameters
        ----------

        Returns
        -------
        """
        for sample in self.train_data:
            if sample["intent"] == self.target_intent:
                for slot_name in sample["slots"]:
                    self.target_intent_slot_names.add(slot_name)

        for i, question in enumerate(self.test_questions):
            if self.test_answers[i]["intent"] == self.target_intent:
                self.target_intent_questions.append(question)

    def _get_artists(self, question) -> Optional[str]:

        prefixes = ["add", "put", "from", "by"]
        suffixes = ["to", "in", "on", "onto"]

        pattern = (
            r"\b("
            + "|".join(prefixes)
            + r")\b ([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*) \b("
            + "|".join(suffixes)
            + r")\b"
        )

        stop_words = [
            "by",
            "the",
            "my",
            "called",
            "current",
            "playlist",
            "album",
            "tune",
            "song",
            "newest",
            "this",
            "onto",
            "my",
            "running",
            "from",
        ]

        match = re.search(pattern, question, re.IGNORECASE)

        if match:
            ans = match.group(2)
            ans = ans.strip().split()

            while ans and ans[0] in stop_words:
                ans.pop(0)
            while ans and ans[-1] in stop_words:
                ans.pop()

            if ans:
                return " ".join(ans)
        else:
            return None

    def _get_playlist(self, question) -> Optional[str]:
        stop_words = ["the", "my", "called", "current", "playlist"]
        pattern = r"(?:(?:\bto\b\s+(.*?)\s+playlist\b)|(?:\bplaylist\b\s+(.*?)\s*))|\b(?:to|onto|in|from|playlist)\s+([^\,]+)"

        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            ans = match.group(1) or match.group(2) or match.group(3)
            ans = ans.strip().split()

            while ans and ans[0] in stop_words:
                ans.pop(0)
            while ans and ans[-1] in stop_words:
                ans.pop()

            if ans:
                return " ".join(ans)

        return None

    def _get_music_item(self, question) -> Optional[str]:
        """
        get by keyword, e.g. track, album, etc.
        """
        keywords = ["track", "album", "tune", "song", "artist"]
        for keyword in keywords:
            if keyword in list(map(lambda x: x.lower(), question.split())):
                return keyword

        return None

    def _get_entity_name(self, question) -> Optional[str]:
        """
        entity name is the song to add to the playlist
        e.g.
            add ... to ...
            add the name ... to ...
            put ... in ...
            put ... on...
            include... in ...
        """
        prefixes = ["add", "put", "include"]
        suffixes = ["to", "in", "on"]

        pattern = (
            r"\b("
            + "|".join(prefixes)
            + r")\b ([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*) \b("
            + "|".join(suffixes)
            + r")\b"
        )

        match = re.search(pattern, question, re.IGNORECASE)

        if match:
            return match.group(2)
        else:
            return None

    def _get_playlist_owner(self, question) -> Optional[str]:
        if "my" in list(map(lambda x: x.lower(), question.split())):
            return "my"

        return None

    def predict_slot_values(self, question):
        """
        Predicts the values for the slots of the target intent.

        Parameters
        ----------
        question: str
                The question for which the slots are to be predicted.

        Returns
        -------
        slots: dict
                The predicted slots.
        """
        get = {
            "playlist": self._get_playlist,
            "music_item": self._get_music_item,
            "entity_name": self._get_entity_name,
            "playlist_owner": self._get_playlist_owner,
            "artist": self._get_artists,
        }

        slots = {}

        for slot_name in self.target_intent_slot_names:
            slots[slot_name] = get[slot_name](question)

        return slots

    def get_confusion_matrix(self, slot_prediction_function, questions, answers):
        """
        Find the true positive, true negative, false positive, and false negative examples with respect to the prediction
        of a slot being active or not (irrespective of value assigned).

        Parameters
        ----------
        slot_prediction_function: Callable
                The function used for predicting slot values.
        questions: list
                The test questions
        answers: list
                The ground-truth test answers

        Returns
        -------
        tp: dict
                The indices of true positive examples are listed for each slot
        fp: dict
                The indices of false positive examples are listed for each slot
        tn: dict
                The indices of true negative examples are listed for each slot
        fn: dict
                The indices of false negative examples are listed for each slot
        """
        tp = {}
        fp = {}
        tn = {}
        fn = {}

        for slot_name in self.target_intent_slot_names:
            tp[slot_name] = []
        for slot_name in self.target_intent_slot_names:
            fp[slot_name] = []
        for slot_name in self.target_intent_slot_names:
            tn[slot_name] = []
        for slot_name in self.target_intent_slot_names:
            fn[slot_name] = []

        for i, question in enumerate(questions):
            pred = slot_prediction_function(question)
            answer = answers[i]["slots"]

            for slot in self.target_intent_slot_names:
                slot_actually_active = slot in answer and answer[slot] is not None
                slot_predicted_active = slot in pred and pred[slot] is not None

                if slot_actually_active and slot_predicted_active:
                    tp[slot].append(i)
                if not slot_actually_active and slot_predicted_active:
                    fp[slot].append(i)
                if not slot_actually_active and not slot_predicted_active:
                    tn[slot].append(i)
                if slot_actually_active and not slot_predicted_active:
                    fn[slot].append(i)

        return tp, fp, tn, fn

    def evaluate_slot_prediction_recall(self, slot_prediction_function):
        """
        Evaluates the recall for the slot predictor. Note: This also takes into account the exact value predicted for the slot
        and not just whether the slot is active like in the get_confusion_matrix() method

        Parameters
        ----------
        slot_prediction_function: Callable
                The function used for predicting slot values.

        Returns
        -------
        accs: dict
                The recall for predicting the value for each slot.
        """
        correct = Counter()
        total = Counter()
        # predict slots for each question
        for i, question in enumerate(self.target_intent_questions):
            i = self.test_questions.index(
                question
            )  # This line is added after the assignment release
            gold_slots = self.test_answers[i]["slots"]
            predicted_slots = slot_prediction_function(question)
            for name in self.target_intent_slot_names:
                if name in gold_slots:
                    total[name] += 1.0  # type:ignore
                    if (
                        predicted_slots.get(name, None) != None
                        and predicted_slots.get(name).lower()
                        == gold_slots.get(name).lower()
                    ):  # This line is updated after the assignment release
                        correct[name] += 1.0  # type:ignore
                    # elif name == "artist":
                    #     print(
                    #         f"{name}, {question}, {gold_slots[name]}, {slot_prediction_function(question)[name]}"
                    #     )
        accs = {}
        for name in self.target_intent_slot_names:
            accs[name] = (correct[name] / total[name]) * 100
        return accs


#####------------- CODE TO TEST YOUR FUNCTIONS
#
# # Define your semantic parser object
# semantic_parser = SemanticParser()
# # Load semantic parser data
# semantic_parser.load_data()
#
# # Evaluating the keyword-based intent classifier.
# # In our implementation, a simple keyword based classifier has achieved an accuracy of greater than 65 for each intent
# print("------------- Evaluating keyword-based intent classifier -------------")
# accs = semantic_parser.evaluate_intent_accuracy(
#     semantic_parser.predict_intent_using_keywords
# )
# for intent in accs:
#     print(intent + ": " + str(accs[intent]))
#
# # Evaluate the logistic regression intent classifier
# # Your intent classifier performance will be 100 if you have done a good job.
# print("------------- Evaluating logistic regression intent classifier -------------")
# semantic_parser.train_logistic_regression_intent_classifier()
# accs = semantic_parser.evaluate_intent_accuracy(
#     semantic_parser.predict_intent_using_logistic_regression
# )
# for intent in accs:
#     print(intent + ": " + str(accs[intent]))
#
# # Look at the slots of the target intent
# print("------------- Target intent slots -------------")
# semantic_parser.get_target_intent_slots()
# print(semantic_parser.target_intent_slot_names)
#
# # Evaluate slot predictor
# # Our reference implementation got these numbers on the validation set. You can ask others on Slack what they got.
# # playlist_owner: 100.0
# # music_item: 100.0
# # entity_name: 16.666666666666664
# # artist: 14.285714285714285
# # playlist: 52.94117647058824
# print("------------- Evaluating slot predictor -------------")
# accs = semantic_parser.evaluate_slot_prediction_recall(
#     semantic_parser.predict_slot_values
# )
# for slot in accs:
#     print(slot + ": " + str(accs[slot]))
#
# # Evaluate Confusion matrix examples
# print("------------- Confusion matrix examples -------------")
# tp, fp, tn, fn = semantic_parser.get_confusion_matrix(
#     semantic_parser.predict_slot_values,
#     semantic_parser.test_questions,
#     semantic_parser.test_answers,
# )
# print(tp)
# print(fp)
# print(tn)
# print(fn)
