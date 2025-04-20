from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

chat_model = ChatOpenAI()

#schema for data format

class Review(TypedDict):
    summary : str
    sentiment : str
    
structured_chat_model = chat_model.with_structured_output(Review)

result = structured_chat_model.invoke('''I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow''')
    
print(result)
print(result['summary'])
print(result['sentiment'])

'''
Here we never mentioned about the summary and sentiment in prompt we just gave review but since we are passing structured output
the prompt is genarated at backend like I want summary and sentiment and then review gets attached that how sturctured output help.

Now in case you dont just want to pass summary and sentiments but needs to give more detailed information then we can pass annotation
as well.
'''