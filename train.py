import argparse
import os
import random
import re
import sys
from collections import defaultdict, deque
from typing import Callable

import dill
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import BaggingRegressor


class Model:
    def __init__(self, path: str = None, n: int = 2) -> None:
        self.n: int = n
        self.__ngramm: defaultdict[tuple[str, ...],
                                   set[str]] = defaultdict(lambda: set())
        self.__vectors: Word2Vec = None

        if path is None:
            self.fit(sys.stdin.read())
        else:
            self.fit_directory(path)

        with open("model.pkl", "rb") as md:
            self.model: BaggingRegressor = dill.load(md)

    def fit_directory(self, path: str) -> None:
        for r, _, f in os.walk(path):
            for file in f:
                self.fit_file(os.path.join(r, file))

    def fit_file(self, path: str) -> None:
        with open(path, "r", encoding="UTF-8") as fl:
            self.fit(fl.read())

    def fit(self, text: str) -> None:
        sents = [tokenize_sents(x) for x in tokenize_text(text)]

        if self.__vectors is None:
            self.__vectors = Word2Vec(sents, min_count=1)
        else:
            self.__vectors.build_vocab(sents, update=True)

        for sent in sents:
            self.__add_sentence(sent)

    def __add_sentence(self, sent: list[str]) -> None:
        for context, word in Ngramm.ngramms(sent, self.n):
            self.__ngramm[context].add(word)

    def generate(self, prefix: str = None, target_length: int = 10) -> str:

        sentence: list[str] = []
        length: int = 0
        res: list[list[str]] = []

        if prefix is not None:
            sents = [tokenize_sents(x) for x in tokenize_text(prefix)]
            sentence = sents[-1]
            res = sents[:-1]

        q: deque[str] = deque(sentence[-self.n:])

        while True:
            if target_length <= length + len(sentence):
                res.append(sentence)
                break

            context: tuple[str, ...] = tuple(q)

            if context in self.__ngramm:
                word_list = list(self.__ngramm[context])
                vec_context = np.concatenate(
                    [self.__vectors.wv[word] for word in context])

                y = [np.concatenate([vec_context, self.__vectors.wv[word]])
                     for word in word_list]

                prediction = self.model.predict(y)
                new_word_idx = max(range(len(prediction)),
                                   key=lambda i: prediction[i])
                new_word = word_list[new_word_idx]
                sentence.append(new_word)
                q.append(new_word)
                q.popleft()
            else:
                if sentence:
                    res.append(sentence)
                    length += len(sentence)
                sentence = list(random.choice(list(self.__ngramm.keys())))
                q.clear()
                q.extend(sentence)

        return ". ".join(map(lambda x: " ".join(x).capitalize(), res)) + "."


class Ngramm:
    @staticmethod
    def ngramms(lst: list[str], n: int = 2) -> \
            list[tuple[tuple[str, ...], str]]:
        res: list[tuple[tuple[str, ...], str]] = []
        q: deque[str] = deque(lst[:n])

        for i in range(n, len(lst)):
            res.append((tuple(q), lst[i]))
            q.popleft()
            q.append(lst[i])
        return res


def split_by(ch: str):
    def delete_empty(func: Callable) -> Callable:
        def wrapper(text: str) -> list[str]:
            return list(filter(len, func(text).split(ch)))
        return wrapper
    return delete_empty


@split_by(".")
def tokenize_text(text: str) -> str:
    return re.sub("[?!;:\n\t]", ".", text).lower()


@split_by(" ")
def tokenize_sents(text: str) -> str:
    return re.sub('[^a-zA-Z\']', " ", text)


def main():
    n: int = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str,
                        help="Path to the input directory")
    parser.add_argument("--model", type=str, help="Model output path")
    args = parser.parse_args()
    mdl = Model(args.input_dir, n)
    with open(args.model, "wb") as fl:
        dill.dump(mdl, fl)


if __name__ == "__main__":
    main()
