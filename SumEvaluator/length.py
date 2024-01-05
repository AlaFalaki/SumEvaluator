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
        return tokenize.tokenize(text, self._stemmer)


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