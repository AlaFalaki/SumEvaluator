import six
import abc
from nltk.stem import porter


# ========================
# Extract the n-grams
# ========================
def create_ngrams(tokens, n):
    """Creates ngrams from the given list of tokens.
    Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.
    Returns:
    A dictionary mapping each bigram to the number of occurrences.
    """

    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
        
    return ngrams


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
        return tokenize.tokenize(text, self._stemmer)


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

        
# =====================
# Calculate the ngrams
# =====================
def calc_novels_by_sum(article_ngrams, summary_ngrams):
    not_seen = 0
    for ngram in six.iterkeys(summary_ngrams):
        if article_ngrams[ngram] == 0:
            not_seen += 1
    
    if len( summary_ngrams ) == 0:
        return 0
    
    return not_seen / len( summary_ngrams )


# ==============================================
# Dictionary to Translate the ngram int to name
# ==============================================
ngram_names_dic = {
    1: 'unigram',
    2: 'bigram',
    3: 'trigram'
}


# ======================
# Calculate the novelty
# ======================
def calculate(articles, summaries, ngrams):
    
    # Make sure the arguments are of type `List`
    if not isinstance(articles, list): raise TypeError("The articles argument must be of type list.")
    if not isinstance(summaries, list): raise TypeError("The summaries argument must be of type list.")
    if not isinstance(ngrams, list): raise TypeError("The ngrams argument must be of type list.")
    
    assert len(articles) == len(summaries), "Articles and Summaries must have the same number of elements."
    
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False)
    
    res = {}
    
    for ngram in ngrams:
        by_summary_avg = average_meter()

        for article, summary in zip(articles, summaries):
            article_tokenized = tokenizer.tokenize( article )
            summary_tokenized = tokenizer.tokenize( summary )

            article_ngrams = create_ngrams( article_tokenized, ngram )
            summary_ngrams = create_ngrams( summary_tokenized, ngram )

            novels_by_sum = calc_novels_by_sum(article_ngrams, summary_ngrams)

            by_summary_avg.update( novels_by_sum )
        
        res[ ngram_names_dic[int(ngram)] ] = by_summary_avg.avg
    
    return res