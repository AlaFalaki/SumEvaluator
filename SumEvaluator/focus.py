import os
import torch
import nltk
from sentence_transformers import SentenceTransformer, util

from IPython.core.display import display, HTML
from tqdm import tqdm
import torch
import pandas as pd

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

# ===============================
# Scale the scores to [0,1] range
# ===============================
def scale(x, min_, max_):
    return (x - min_) / (max_ - min_)

# =======================
# Step 1:
#    Calculate the scores
# =======================
def prepare(articles, summaries, exp_name="focus_data", model="all-MiniLM-L6-v2"):
    
    # Make sure the arguments are of type `List`
    if not isinstance(articles, list): raise TypeError("The articles argument must be of type list.")
    if not isinstance(summaries, list): raise TypeError("The summaries argument must be of type list.")
    
    assert len(articles) == len(summaries), "Articles and Summaries must have the same number of elements."
    
    os.mkdir(f"./{exp_name}")
    
    device='cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = SentenceTransformer(model, device=device)
    
    scores={}
    lengths = []
    distributions = {}
    
    for i in range( len(articles) ):
        the_article = nltk.sent_tokenize( articles[i] )
        the_summary = [ summaries[i] ] * len( the_article )
        lengths.append( len( the_article ) )

        embeddings1 = model.encode(the_article, convert_to_tensor=True)
        embeddings2 = model.encode(the_summary, convert_to_tensor=True)

        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        for idx in range( len(the_article) ):
            if idx not in scores:
                scores[idx] = average_meter()

            if idx not in distributions: distributions[idx] = 1
            else: distributions[idx] += 1

            scores[idx].update( cosine_scores[idx][idx] )

    torch.save( scores, f"./{exp_name}/scores.pt" )
    torch.save( lengths, f"./{exp_name}/lengths.pt" )
    torch.save( distributions, f"./{exp_name}/distributions.pt" )

# =====================
# Step 2:
#   Show the results.
# =====================
def illustrate(exp_name):
    
    # Load Generated Summaries
    scores = torch.load(f"./{exp_name}/scores.pt", map_location=torch.device("cpu"))
    lengths = torch.load(f"./{exp_name}/lengths.pt", map_location=torch.device("cpu"))
    distributions = torch.load(f"./{exp_name}/distributions.pt", map_location=torch.device("cpu"))
    
    # Get Average Len
    average_len = int( sum(lengths) / len(lengths) )
    lengths_pd = pd.DataFrame(lengths)
    lengths_desc = lengths_pd.describe()
    
    # Get distribution percentage
    twentyfive = int( lengths_desc[0]['25%'] )
    fifty = int( lengths_desc[0]['50%'] )
    seventyfive = int( lengths_desc[0]['75%'] )
    
    # Get distribution percentage
    total_rows = len( lengths )
    for idx, item in distributions.items():
        distributions[idx] = (item/total_rows)
    
    # Scale to [0, 1]
    min_ = 1
    max_ = 0
    for idx, score in scores.items():
        avg = scores[idx].avg

        if avg < min_:
            min_ = avg

        if avg > max_:
            max_ = avg
    
    for idx, score in scores.items():
        scores[idx] = int( scores[idx].avg * 100)
    
    helper = "<br /><div><div style='padding:2px; border:2px solid #f80000; display: inline-block;'>25% of article samples have {} sentences</div> ".format(twentyfive)
    helper += "<div style='padding:2px; border:2px solid #f800e3; display: inline-block'>50% of article samples have {} sentences</div> ".format(fifty)
    helper += "<div style='padding:2px; border:2px solid #ffa400; display: inline-block'>75% of article samples have {} sentences</div></div><br />".format(seventyfive)

    the_html = '<div style="line-height: 2.6rem;">'
    for idx, item in scores.items():
        border=0
        color="FFF"
        if idx+1 == twentyfive:
            border=2
            color="f80000"

        if idx+1 == fifty:
            border=2
            color="f800e3"

        if idx+1 == seventyfive:
            border=2
            color="ffa400"

        the_html += '<div title="{}%" style="background-color: rgba(0, 153, 0, {}); border: {}px solid #{}; margin-left:2px; padding:2px; margin-bottom:5px; display: inline-block;">Sentence-#{}</div> '.format(item, item/100, border, color, idx+1)

    the_html+="</div>"
    
    display(HTML('{} {}'.format(helper, the_html)))