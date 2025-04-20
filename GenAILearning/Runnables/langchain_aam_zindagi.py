import random

# class for creating LLM

class NakliLLM:
    def __init__(self):
        print("LLM is created")
    
    def predict(self, prompt):
        response_list=[
            'Delhi is capital of india',
            'IPL is cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}
    
# langchain class for prompts

class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def format(self, **input_dict):
        return self.template.format(**input_dict)
    
# langchain class create custom function to integrate prompt and llm component

class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm=llm
        self.prompt = prompt
    
    def run(self, input_dict):
        final_prompt = self.prompt.format(**input_dict)
        result = self.llm.predict(final_prompt)
        
        return result['response']
        
    
llm = NakliLLM()

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length','topic']
)


chain = NakliLLMChain(llm, template)

result = chain.run({'length':'short', 'topic': 'india'})

print(result)


'''
We can see here we needed to write custom function to connect llm and prompt. 
'''

'''
more about self keyword:

when we create any new object, self establishes connection between current object being created and class
It referes to current object that is being created. 

__init__(self) is like saying:

"Hey, when someone walks in (an object is created), give me their name so I know who I’m talking to."

If you don’t have self, it’s like:

"I’m a waiter, but I don’t want to know which customer I’m serving..." 🍝😵

when you create new object python passes new object(self) and any attributes if passed to int function.
'''
    