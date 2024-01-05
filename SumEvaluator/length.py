import re
import abc
from nltk.stem import porter

# ========================
# Class to create average 
# ========================
class average_meter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
# ========================
# Tokenizer class
# ========================
class Tokenizer(abc.ABC):
    """Abstract base class for a tokenizer.
    Subclasses of Tokenizer must implement the tokenize() method.
    """

    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
    """Default tokenizer which tokenizes on whitespace."""

    def __init__(self, use_stemmer=False):
        """Constructor for DefaultTokenizer.
        Args:
          use_stemmer: boolean, indicating whether Porter stemmer should be used to
          strip word suffixes to improve matching.
        """
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return tokenize(text, self._stemmer)


# =======================
# Calculate the lenghts
# =======================
def calculate(articles, summaries):
    
    # Make sure the arguments are of type `List`
    if not isinstance(articles, list): raise TypeError("The articles argument must be of type list.")
    if not isinstance(summaries, list): raise TypeError("The summaries argument must be of type list.")
    
    assert len(articles) == len(summaries), "Articles and Summaries must have the same number of elements."

    tokenizer = DefaultTokenizer(use_stemmer=False)
    
    summary_length = average_meter()
    summary_article_ratio = average_meter()

    for article, summary in zip(articles, summaries):
        sum_len = len( tokenizer.tokenize( article ) )
        art_len = len( tokenizer.tokenize( summary ) )

        summary_length.update( sum_len )
        summary_article_ratio.update( art_len/sum_len )
        
    return {"average_summary_length": summary_length.avg,
            "average_article_summary_ratio": summary_article_ratio.avg}



# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library for tokenizing text."""


# Pre-compile regexes that are use often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)


def tokenize(text, stemmer):
    """Tokenize input text into a list of tokens.
    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.
    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
    Returns:
    A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = NON_ALPHANUM_RE.sub(" ", six.ensure_str(text))

    tokens = SPACES_RE.split(text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x
                  for x in tokens]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]

    return tokens