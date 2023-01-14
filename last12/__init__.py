from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных

#пытаюсь добавить sql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
#добавляю некую статику
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


routes = [
    ...
    Mount('/static', app=StaticFiles(directory='static'), name="static"),
]

app = Starlette(routes=routes)

class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str
app = FastAPI()
classifier = pipeline('text-generation', model='gpt2') #classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    """Эта функция вызывает классификатор"""
    return {"message":'bike and car'}
@app.post("/predict/")
def predict(item: Item):
    """проводим генерацию текста"""
    return classifier(item.text)[0]