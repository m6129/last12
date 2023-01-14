from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных


class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str
app = FastAPI()
classifier = pipeline('text-generation', model='gpt2') #модель генерации текста по слову
classifier_two = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") #модель описания изображений

@app.get("/")
def root():
    """Эта функция вызывает классификатор"""
    return {"message":'Добро пожаловать!!!'}
@app.post("/predict/")
def predict(item: Item):
    """проводим генерацию текста по слову"""
    return classifier(item.text)[0]


@app.post("/predict_Z/")
def predict_Z(item: Item):
    """проводим генерацию текста по изображению"""
    return classifier(item.text)[0]

