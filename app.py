import chainlit as cl
import markdown
from bs4 import BeautifulSoup
from rag_functions import RagFunctions

r = RagFunctions()

@cl.on_chat_start
async def on_chat_start():
   await cl.Message(
       content = "Welcome to BioBot, a Biology chat assistant"
   ).send()
    

@cl.on_message
async def on_message(msg : cl.Message):
    query = msg.content

    response = r.generate_response(query)
    response = markdown.markdown(response)
    soup = BeautifulSoup(response , 'html.parser')
    response = soup.get_text()
    await cl.Message(
        content=response
    ).send()