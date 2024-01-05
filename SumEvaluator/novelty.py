import six
from utils import tokenizers
from utils.create_ngrams import create_ngrams
from utils.average_meter import average_meter

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