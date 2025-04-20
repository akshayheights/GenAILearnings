'''
we will get text input based on that using two LLM models we need to genrate notes and text. 
THen using third model we need to combine them.
'''
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

# Runnable parallel is kind of runnable with help of which we can run multiple chains in parallel.

load_dotenv()

chat_model_1 = ChatOpenAI()

parser = StrOutputParser()


prompt_1 = PromptTemplate(
    template='Generate short and simple notes from follwing text \n {text}',
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template='Generate 5 short question and answers \n {text}',
    input_variables=['text']
)

prompt_3 = PromptTemplate(
    template='Merge the provided notes and quiz into one single document \n notes : {notes} \n quiz:{quiz}',
    input_variables=['notes','quiz']
)


parallel_chain = RunnableParallel({
    'notes': prompt_1 | chat_model_1 | parser,
    'quiz': prompt_1 | chat_model_1 | parser 
})

merge_chain = prompt_3 | chat_model_1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text})

print(result)

print(chain.get_graph().print_ascii())



