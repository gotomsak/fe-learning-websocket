import asyncio
import websockets
import io
from PIL import Image
from HumanConditionDetection import HumanConditionDetection
from FaceLandmark import FaceLandmark
import json
import mariadb
import sys
from dotenv import load_dotenv
import os

# from GetFrequency import GetFrequency
# hcd = HumanConditionDetection()
load_dotenv()




async def server(websocket, path):
    fl = FaceLandmark(0.25, 0.25)
    cnt = 0
    # freq = GetFrequency(0.25, 0.25)
    
    while True:
        name = await websocket.recv()
        # print(name)
        print(type(name))
        # for i in name:
        #     img_binarystream = io.BytesIO(i)
        #     img_pil = Image.open(img_binarystream)
        # print(name[0:5])
        if "base64" not in name:
            continue
        
        res = fl.main_face_landmark(name, 1.2, 0.5, 1.2, 0.5)
    # for i in name:
    #     print(i)
        # print(f"< {name}")
        # await asyncio.sleep(0.5)
    # greeting = f"Hello {name}!"
        res = json.dumps(res)
        await websocket.send(res)
    # print(f"> {greeting}")



start_server = websockets.serve(server, "localhost", 8765)
print("start")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
