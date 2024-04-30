import asyncio
from typing import AsyncIterable
import os
import requests
import json
import random
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"]=''

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    website_type: str
    content_type: str
    expert_in: str="website content creation"
    length:int=50

# class ImageType(BaseModel):
#     image_of: str
#     orientation: bool
#     size: str


# async def send_message(content: str) -> AsyncIterable[str]:
#     callback = AsyncIteratorCallbackHandler()
#     model = ChatOpenAI(
#         streaming=True,
#         verbose=True,
#         callbacks=[callback],
#     )
async def send_message(message: Message) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        temperature=0.7
    )

    # task = asyncio.create_task(
    #     model.agenerate(messages=[[HumanMessage(content=content)]])
    # )

    template = (
    "You are expert in {expert_in}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Create content for {content_type} section of {length} words for a {website_type} website"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
    )
    complete_prompt=  chat_prompt.format_prompt(
        expert_in=message.expert_in, content_type=message.content_type, length=message.length, website_type=message.website_type
    ).to_messages()   #complete_prompt is an array of systemprompt and HumanPrompt


    task = asyncio.create_task(
        
        # model.agenerate(messages=[[HumanMessage(content=message.website_type)]])
        model.agenerate(messages=[complete_prompt])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


# @app.post("/stream_chat/")
# async def stream_chat(message: Message):
#     generator = send_message(message.content)
#     return StreamingResponse(generator, media_type="text/event-stream")

@app.post("/text_content")
async def stream_chat(message: Message):
    generator = send_message(message)
    # return message
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/image/")
async def get_image(image_name:str, orientation: str="landscape", size: str="original", per_page:int=1):

    # image_name="red flowers"
    # orientation="landscape"
    # per_page=3
    # size="small"
    page_no=random.randint(1,10)
    
    pexel_api="KYA5omO4oxYPmua1IRniIB1iDZiAJmubzQ5xHOT3w770K330iHkXUM19"
    # https://api.pexels.com/v1/search?query=red car&orientation=landscape&per_page=1&size=small&page=page_no
    # keyword="red car"
    url=f"https://api.pexels.com/v1/search?query={image_name}&orientation={orientation}&per_page={per_page}&size={size}&page={page_no}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
        "Authorization": "KYA5omO4oxYPmua1IRniIB1iDZiAJmubzQ5xHOT3w770K330iHkXUM19"
    }

    # headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers)
    byte_response_content=response.content
    json_data = json.loads(byte_response_content.decode('utf-8'))

    photos=json_data["photos"]
    if size=="original":
        required_photos=[photo["src"][orientation] for photo in photos]
    else:
        required_photos=[photo["src"][size] for photo in photos]

    return json.dumps(required_photos)

    # https://api.pexels.com/v1/search?query=car&orientation=landscape"
    # if message.islandscape:
    # orientation=""
    # headers = {"Content-type": "application/json"}
    # with requests.post(url, json=data, headers=headers)
