from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text='''
# Define a simple class
class Car:
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

    def start(self):
        print(f"The {self.color} {self.brand} is starting.")

# Create an object of the class
my_car = Car("Toyota", "red")

# Call a method on the object
my_car.start()

'''

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

splitted_text = text_splitter.split_text(text)

print(splitted_text[0])