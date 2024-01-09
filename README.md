
SumEvaluator
===

This library offers an array of supplementary tools designed to enhance the traditional ROUGE score used in assessing generated summaries. Included in this package are four distinct metrics aimed at evaluating models in terms of Length, Novelty, Focus, and GPT-4 score.

## Usage

Run the following command to install the package.

```
pip install -q git+https://github.com/AlaFalaki/SumEvaluator.git
```

You can see usage example in the [demo.ipynb notebook].(https://colab.research.google.com/github/AlaFalaki/SumEvaluator/blob/main/demo.ipynb).

## Metrics


#### GPT-4 Evaluation
This metric will use the GPT-4 model like a human evaluator and assign a score to the generated summary with respect to the article. (instead of the target summary) The approach returns four scores for Relevance, Consistency, Fluency, and Coherence. It will evaluate the generated summary on different aspects, such as including important information from the article or grammatical correctness. Refer to the defined prompt in the [gpt.py](https://github.com/AlaFalaki/SumEvaluator/blob/main/SumEvaluator/gpt.py#L8) file for a full definition of each metric.

```
import SumEvaluator

SumEvaluator.gpt.calculate( [ARTICLE],
                            [SUMMARY],
                            api_key="<OpenAI_API_KEY>")

# {'relevance': 9.0, 'consistency': 9.0, 'fluency': 10.0, 'coherence': 9.0}
```

You could either pass the API key directly to the `.calculate()` method, or set the `OPENAI_API_KEY` key in your Python environment.

___

#### Novety
This metric calculates the number of n-grams in the summary but is absent in the article. It is useful for assessing the extent of the model's replication behaviour.

```
SumEvaluator.novelty.calculate(articles=[ARTICLE],
                               summaries=[SUMMARY],
                               ngrams=[1, 2])

# {'unigram': 0.23809523809523808, 'bigram': 0.5833333333333334}
```

___

#### Focus Finder
This visualization method employs an embedding model to determine the cosine similarity score between the generated summary and each sentence from the article. Then, the `illustrate()` method helps with illustrating where the model is concentrating its attention during the summary generation process.

```
SumEvaluator.focus.prepare([ARTICLE], [SUMMARY], "test_proj")
SumEvaluator.focus.illustrate("test_proj")
```

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/AlaFalaki/SumEvaluator/main/images/focus_finder.png" />
</p>

___

#### Length
A simple metric that measure the average length of the generated summaries and the average ratio of them with respect to the articles lenghts.


```
SumEvaluator.length.calculate([ARTICLE], [SUMMARY])

# {'average_summary_length': 49.0, 'average_article_summary_ratio': 7.73469387755102}
```
## Requirements

* Python 3.8.10
* torch 1.10.0
* nltk 3.8.1
* sentence_transformers 2.2.2
* pandas 1.5.3
* openai 0.28
* six 1.16.0

## Citations
If you wish to cite the paper, you may use the following:
```
@paper comming soon?
```

GL!