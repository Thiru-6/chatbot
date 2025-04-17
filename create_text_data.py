import pandas as pd

dataset = pd.read_csv("train.csv")

i = 0
text_no = 1
while i < len(dataset):

    if i + 10000 > len(dataset):
        text_data = [dataset["inputs"][i : ]]
    else:
        text_data = [dataset["inputs"][i : i + 10000]]
    i += 10000
    file_name = f"data{text_no}.txt"
    file = open(file_name , "w" , encoding="utf-8")
    for text in text_data:
        file.writelines(text)
    file.close()
    text_no = text_no + 1