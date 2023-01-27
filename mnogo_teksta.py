#https://github.com/chetverovod/webfasumgas/edit/main/webfasumgaz.py Игорь сделал проект который работает с большими объёмами текста 
#на fastapi, по крайне мере через терминальный запрос
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Item(BaseModel):
    text: str

app = FastAPI()

model_name = "IlyaGusev/rugpt3medium_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#-----------------------------------------------------------------
def summarize(model, tokenizer, article_text):
    """Function makes and returns summary of article_text.
 
  Parameters
  ----------
     model:
          Model object.
     tokenizer:
          Tokens generator.
      article_text : str
          Input text.

  Returns
  -------
      str
          Summary of input text.
   """

    text_tokens = tokenizer(
      article_text,
      max_length = 600,
      add_special_tokens = False, 
      padding = False,
      truncation = True
      )["input_ids"]

    input_ids = text_tokens + [tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids])

    output_ids = model.generate(
      input_ids=input_ids,
      no_repeat_ngram_size = 4,
      max_new_tokens = 100
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens = False)
    summary = summary.split(tokenizer.sep_token)[1]
    summary = summary.split(tokenizer.eos_token)[0]
    return summary  

#-----------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Webfasumgaz application is online!"}

#-----------------------------------------------------------------
@app.post("/predict/")
def predict(item: Item):
  """Method builds summary for article from newpaper. 
  Model "IlyaGusev/rugpt3medium_sum_gazeta" is used.
 
  Parameters
  ----------
  item: str
         Input text of article.
  
  Returns
  -------
  str
     Summary of article.
  """
  
  txt = item.text 
  res = "Error: text is absent."
  if txt:
     sum = summarize(model, tokenizer, txt)
     res = sum 
  return res    
