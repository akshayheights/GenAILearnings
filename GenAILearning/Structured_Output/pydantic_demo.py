from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'nitish' # setting default values
    age : Optional[int] = None # optional values
    email : EmailStr # default validation
    cgpa : float = Field(gt=0, lt=10, default=5, description="decimal value representing cgpa of student") # applying constraint
    
    
new_student = {"name":"akshay", "age":20, 'email':'abc@gmail.com', 'cgpa':9}

student = Student(**new_student)

print(student) # this is pydantic object

student_dict = dict(student)

student_json = student.model_dump_json()

print(student_json)

print(type(student.name))

print(student_dict['age'])

'''
even if we pass age as string pydantic is able to do implicit conversion in backend to convert it into int.
it is known as type coercing.
'''

