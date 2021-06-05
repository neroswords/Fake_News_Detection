# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: NishitP
"""

import pickle
import tkinter as tk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import DataPrep
#doc_new = ['obama is running for president in 2016']

H = 600 # กำหนดค่าตัวแปรความสูงหน้าต่างโปรแกรม
W = 800 # ตัวแปรความกว้าง

load_logR_pipeline_ngram_model = pickle.load(open('final_model.sav', 'rb'))
load_LSTM_model = load_model('LSTM_model.h5')
load_Embedding_model = load_model('Embedding_model.h5')

root = tk.Tk()

canvas = tk.Canvas(root, height=H, width=W)
canvas.pack()

frame = tk.Frame(root, bg='#80c1ff', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

entry = tk.Entry(frame, font=('Courier', 18))
entry.place(relwidth=0.65, relheight=1)

button = tk.Button(frame, text='ตกลง', font=40, command=lambda: detecting_fake_news(entry.get()))
button.place(relx=0.7,  relwidth=0.1, relheight=1)

button = tk.Button(frame, text='reset', font=40, command=lambda: delete_text())
button.place(relx=0.85, relwidth=0.1, relheight=1)

lower_frame = tk.Frame(root, bg= '#80c1ff', bd=5)
lower_frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')

label = tk.Label(lower_frame, font=('Courier', 18), anchor='nw', justify='left')
label.place(relwidth=1, relheight=1)

# var = input("Please enter the news text you want to verify: ")
# print("You entered: " + str(var))

def delete_text():
    entry.delete(0,tk.END)
    return
#function to run for prediction
def detecting_fake_news(var):    
#retrieving the best model for prediction call
    t= Tokenizer()
    t.fit_on_texts(DataPrep.train_news['Statement'].values)
    vocab_size = len(t.word_index)+1
    encoded_docs = t.texts_to_sequences([var])
    padded_docs = pad_sequences(encoded_docs, maxlen=4, padding="post")

    new_prediction = (load_LSTM_model.predict(padded_docs)>= 0.5).astype(int)
    prediction = load_logR_pipeline_ngram_model.predict([var])
    prob = load_logR_pipeline_ngram_model.predict_proba([var])
    answer = "The given statement is %s \nThe truth probability score is %f\nAnd new predict is %d"%(prediction[0],prob[0][1],new_prediction)
    label['text'] = answer
    # (print("The given statement is ",prediction[0]),
    #     print("The truth probability score is ",prob[0][1]))


if __name__ == '__main__':
    root.mainloop()
    # detecting_fake_news(var)