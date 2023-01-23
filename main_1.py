from fastapi import FastAPI #импортируем библиотеку для API
from transformers import pipeline #импортируем библиотеку для использования моделей с hugginface
from pydantic import BaseModel #pydantic, занимается автоматической проверкой формата и типа данных
from fastapi.staticfiles import StaticFiles#для загрузки статических файлов
import mimetypes #MIME-типы — типы данных, которые могут быть переданы посредством сети Интернет с применением стандарта MIME
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse, FileResponse #позовляет получать ответы в других форматах 
#https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
from fastapi import UploadFile, File #для загрузки файлов https://fastapi.tiangolo.com/tutorial/request-files/  https://itnan.ru/post.php?c=1&p=710376

import spacy #библиотека для NLP и NER
from spacy import displacy #библиотека для создания html файлов 


class Item(BaseModel):# импортируем из pydantic базовый вариант класса для моделей BaseModel и создаем на его основе свою модель:
    text: str #получать на ввод строковые данные
app = FastAPI() # 

app.mount("/static", StaticFiles(directory="public", html=True)) #создаём html страницу, которую можно просмотреть по адресу: #http://127.0.0.1:8000/static/index.html

#обозначаем модели из HugginFace
classifier = pipeline('text-generation', model='gpt2') #модель генерации текста по слову
classifier_two = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") #модель описания изображений
#некоторый код для работы модели spaCy
nlp = spacy.load("en_core_web_lg") #выбираем модель для NER spaCy
doc = nlp('test')
 
#выбираем приветствие https://colorscheme.ru/html-colors.html https://htmled.it/redaktor/
@app.get("/")#корневая папка
def root():
    data = """<h2 style="text-align: center; color: #008000;">Добро пожаловать!!!!</h2> 
    <h3 style="text-align: center; color: #00CED1;">Это итоговый проект группы №12</h3>
    <h4 style="text-align: center; color: #FF4500;">Исполнители: Зайцев Антон Александрович, Чурилов Алексей Александрович, Зайцев Александр Васильевич</h4>"""
    return HTMLResponse(content=data) #возвращаем наше замечатльное приветствие

def spa(Item):
    html_e = displacy.render(Item, style='ent') #а этот же код уже выполняется, до того как мы вводим текст 
    with open('data_ent.html',"w") as f:
        f.write(data_ent.html)
    return "public/data_ent.html"

#блок работы с моделями было (app.get)
@app.post("/file", response_class = FileResponse)
def root_html(item: Item): #прописываем. что подаём на вход
    return spa(item)#"/public/data_ent.html"
    #return "<h2>Hello METANIT.COM</h2>"

@app.post("/predict/")
def predict(item: Item):
    """проводим генерацию текста по слову"""
    return classifier(item.text)[0]


@app.post("/predict_Z/")
def predict_Z(item: Item):
    """проводим генерацию текста по изображению"""
    return classifier_two(item.text)[0]

@app.post("/predict_s/")
def predict_s(item: Item):
    """анализ новостей"""
    #html = displacy.render(doc, style='dep')
    #with open('data_graf.html',"w") as f:
    #    f.write(html)
    #html
        
    html_e = displacy.render(doc, style='ent')
    with open('data_ent.html',"w") as f:
        f.write(html_e)
    html_e
    return doc(item.text)[0]
    #app.mount("/static", StaticFiles(directory="public", html=True))


    #return FileResponse("public/index.html", 
                        #filename="mainpage.html", 
                        #media_type="application/octet-stream")

