from utils.average_meter import average_meter
from utils import tokenizers

# =======================
# Calculate the lenghts
# =======================
def calculate(articles, summaries):
    
    # Make sure the arguments are of type `List`
    if not isinstance(articles, list): raise TypeError("The articles argument must be of type list.")
    if not isinstance(summaries, list): raise TypeError("The summaries argument must be of type list.")
    
    assert len(articles) == len(summaries), "Articles and Summaries must have the same number of elements."

    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False)
    
    summary_length = average_meter()
    summary_article_ratio = average_meter()

    for article, summary in zip(articles, summaries):
        sum_len = len( tokenizer.tokenize( article ) )
        art_len = len( tokenizer.tokenize( summary ) )

        summary_length.update( sum_len )
        summary_article_ratio.update( art_len/sum_len )
        
    return {"average_summary_length": summary_length.avg,
            "average_article_summary_ratio": summary_article_ratio.avg}