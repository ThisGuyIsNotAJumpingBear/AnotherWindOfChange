# LLM for LSC Detection

This is the official implemention for work `Large Language Models on Lexical Semantic Change Detection: A Comprehensive Evaluation`.

## Prerequisites
1. Create your virtual environment 
`conda create -n 2611final python=3.11`
2. Install packages
`pip install -r requirements.txt`

### Running Codes
#### PPMI
Run `ppmi.ipynb`.
#### SGNS
Run `sgns.ipynb`.
#### BERT
Run `bert.ipynb`. We recommend running this on a gpu as we initialize a hugging face model.
#### LLMs
We use multiprocessing to parallelize the api requests to openai. Code for this can be found in `gpt.py`. However, we do not recommend running this file as you would need to create your own `.env` file, and change the openai organization id to your own in line `13` of `gpt.py`. All responses from gpt4 have been saved in `/data` under `gpt4_answers_no_date.pkl`, `gpt4_answers_qiq.pkl`, `gpt4_answers_with_date.pkl`.

To run the evaluations of these models, run `gpt.ipynb`.

We ran out of time for introducing chain-of-thought(cot) into our paper but we do have cot responses from GPT-3.5 which are parsed and can be found in `cot_repsonses.txt`. 
