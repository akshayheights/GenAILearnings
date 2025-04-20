from abc import ABC, abstractmethod
import random

# Abstract base class
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

# Fake LLM
class NakliLLM(Runnable):
    def __init__(self):
        print("LLM is created")
    
    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}
    
    def predict(self, prompt):
        # This isn't used currently but could be used in another context
        return self.invoke(prompt)

# Prompt template
class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)

# Output parser (extracts 'response' from dict)
class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']

# Connector: connects all runnables in a pipeline
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)  # ✅ FIXED
        return input_data

# --- Instantiate components ---
llm = NakliLLM()

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

parser = NakliStrOutputParser()

# Build pipeline
chain = RunnableConnector([template, llm, parser])

# Run it
result = chain.invoke({'length': 'long', 'topic': 'india'})
print("Final Output:", result)

'''
here we are passing input_data as parameter to object of runnableconenctor class for method invoke.
this method internally class invoke method of each object such as template, llm and parser. and output of one function it sends to next funtion.
since we are passing object as list invoke method gets called for particular object.

We can add more object i.e. more component and our runnable_connector class will take care of connecting them.
'''
