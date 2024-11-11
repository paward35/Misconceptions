<h1>Misconceptions</h1>

<h2>Steps</h2>
<ol>  
<li>~~ Generate Prompts ~~</li>  
<li>Get Misconceptions</li>  
<li>Similarity Matching</li>  
<li>Generate + Save Results</li>  
</ol>


## Repo structure
/data -> Data files 
/notebooks
-- data_sandbox -> notbook to explore data creator
-- PrepareFeatures -> Paddy's exploratory notebbok
data.py -> Main dataset creator. Configurable to create different variations of fine-tuning datasets.
utils.py -> utility functions (apk, mapk[25]) 
train.py -> non implemented train code, will follow this general structure for finetuning: https://github.com/SeanGormann/llm_recovery/blob/main/dpo_main.ipynb 
test.py -> non implemented code to evaluate performance 


## Example data point:

instruction: 'Here is a question for you:\nSimplify the following, if possible: \\( \\frac{m^{2}+2 m-3}{m-3} \\)\nThe correct answer is: Does not simplify\nHowever, a common misconception is: \\( m+1 \\)\nThis question is related to: Simplify an algebraic fraction by factorising the numerator.\nSubject area: Simplifying Algebraic Fractions.',

chosen_response: 'Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term\n',

rejected_response: 'Believes interior angles are outside a shape',

chosen_score: 1
rejected_score: 0