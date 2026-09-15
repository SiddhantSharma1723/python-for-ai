from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Student(BaseModel):
    hours_studied:int
    attendance:float
    previous_score:float

@app.post("/student")
def student_data(student:Student):
    return student