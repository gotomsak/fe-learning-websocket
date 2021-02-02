import asyncio
import websockets
import io
from FaceLandmark import FaceLandmark
import json
import sys
import os
import ssl
# from GetFrequency import GetFrequency
# hcd = HumanConditionDetection()


async def server(websocket, path):
    fl = FaceLandmark(0.25, 0.25)
    cnt = 0
    # freq = GetFrequency(0.25, 0.25)

    while True:
        name = await websocket.recv()

        print(type(name))

        if "base64" not in name:
            fl.save_dir_path = name
            fl.create_dir()
            continue

        res = fl.main_face_landmark(name)

        res = json.dumps(res)
        await websocket.send(res)

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('./ssl/fullchain.pem', './ssl/privkey.pem')
#start_server = websockets.serve(server, "gotomsak", 8765, ssl=ssl_context)
start_server = websockets.serve(server, "127.0.0.1", 8765, ssl=ssl_context)
print("start")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
