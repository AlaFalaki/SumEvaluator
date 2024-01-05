import os
import json
import openai

# ==================
# Constant Variables
# ==================
EVAL_PROMPT  = """
Imagine you are a human annotator now. You will evaluate the quality of summaries written for a news article. Please follow the steps:

1. Carefully read the news article, and be aware of the information it contains.
2. Read the proposed summary.
3. Rate the summary on four dimensions: relevance, consistency, fluency, and coherence. You should rate on a scale from 1 (worst) to 10 (best).

Definitions are as follows:
Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
Fluency: This rating measures the quality of individual sentences, whether they are well-written and grammatically correct. Consider the quality of individual sentences.
Coherence: The rating measures the quality of all sentences collectively, to fit together and sound natural. Consider the quality of the summary as a whole.

Lastly, the output must be JUST in JSON format as follows, don't include anything before or after it:
{"explaination": "<explain>", "relevance": <relevance_score>, "consistency": <consistency_score>, "fluency": <fluency_score>, "coherence": <coherence_score>}
Where the <relevance_score>, <consistency_score>, <fluency_score>, and <coherence_score> are the placeholders for the scores based on your evauation, also the <explain> placeholder is where to justify your reasoning for each metric.

The article and the summary are given below:
"""
PRE_ARTICLE = "\nArticle:\n"
PRE_SUMMARY = "\n\nSummary:\n"


def calculate(articles, summaries, api_key, exp_name="gpt_data", model_name="gpt-4-1106-preview"):
    
    # Make sure the arguments are of type `List`
    if not isinstance(articles, list): raise TypeError("The articles argument must be of type list.")
    if not isinstance(summaries, list): raise TypeError("The summaries argument must be of type list.")
        
    assert len(articles) == len(summaries), "Articles and Summaries must have the same number of elements."
    
    os.mkdir(f"./{exp_name}")
    
    if api_key:
        openai.api_key = api_key
        
    # ================
    # Get the Results
    # ================
    res = []
    for idx, (article, summary) in enumerate( zip(articles, summaries) ):
        tmp_prompt = f"{EVAL_PROMPT}{PRE_ARTICLE}{article}{PRE_SUMMARY}{summary}"
        
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Imagine you are a human annotator now. You will evaluate the quality of summaries written for a news article."},
                    {"role": "user", "content": tmp_prompt},
                ],
                temperature=0,
                max_tokens=1024,
                n=1
            )
            response = response['choices'][0]['message']['content']
            if response[0:7] == "```json": # Remove unnecessary indicator that sometimes happens.
                response = response[7:-3]

            response_jsond = json.loads( response )
            res.append( response_jsond )
        
        except Exception as error:
            print( error )
            print(">> error on index", idx)
    
    # ================
    # Save the results
    # ================
    with open(f"./{exp_name}/scores.json", "w") as json_file:
        json.dump(res, json_file, indent=2)
    
    # ============================
    # Calculate the Average Scores
    # ============================
    relevance = []
    consistency = []
    fluency = []
    coherence = []

    for item in res:
        relevance.append( int(item['relevance']) )
        consistency.append( int(item['consistency']) )
        fluency.append( int(item['fluency']) )
        coherence.append( int(item['coherence']) )

    return { "relevance": sum(relevance)/len(relevance),
             "consistency": sum(consistency)/len(consistency),
             "fluency": sum(fluency)/len(fluency),
             "coherence": sum(coherence)/len(coherence) }
    