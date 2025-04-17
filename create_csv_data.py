import dotenv
import os
import pandas as pd
from datasets import load_dataset


def createData(question , context , response):
    inputs = []
    outputs = list(response)
    for i in range(0 , len(context)):
        input_prompt = "Answer the Question based on the context. context : " + context[i] + "Question : " + question[i] 
        inputs.append(input_prompt)

    return inputs , outputs


dotenv.load_dotenv()

hf_token = os.environ["hf_token"]
dataset_name = "fedml/PubMedQA_instruction"

dataset = load_dataset(dataset_name)

train_context = list(dataset["train"]["context"])
train_questions = list(dataset["train"]["instruction"])
train_response = list(dataset["train"]["response"])

test_context = list(dataset["test"]["context"])
test_questions = list(dataset["test"]["instruction"])
test_response = list(dataset["test"]["response"])

train_inputs , train_outputs = createData(train_questions , train_context , train_response)
test_inputs , test_outputs = createData(test_questions , test_context , test_response)


train_data = {
    "inputs" : train_inputs,
    "outputs" : train_outputs
}

test_data = {
    "inputs" : test_inputs,
    "outputs" : test_outputs
}


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")