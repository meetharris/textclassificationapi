from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, Request, status, File, Form, UploadFile, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
@app.get("/items/{item_id}")
async def read_id(item_id: int):
    print(item_id)
    import copy_Recognize_face_image
    copy_Recognize_face_image.compare_faces(item_id)
    return {"item_id": item_id}
'''

@app.post("/")
async def read_id():
    return {"hello"}

@app.get("/myname")
async def read_id(name: str):
    print(name)
    return {"name": name}

@app.post('/get_info/')
async def read_image(file: UploadFile = File(...), item_id: int = Form(...)):
    contents = await file.read()
    print(type(contents))
    #print(item_id)
    # print(file)
    import cv2
    import numpy as np
    decoded_image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
    print(type(decoded_image))
    from Recognize_face_image import compare_faces
    result, execution = compare_faces(item_id, decoded_image)
    execution = str(round(execution, 2))
    print(result)
    print(execution+'s')
    return {'Result': result, 'executionTime': execution + 's'}

@app.post('/predict_Category/')
async def read_csv(file: UploadFile = File(...), item_id: int = Form(...)):
    contents = await file.read()
    print(type(contents))


    #print(item_id)
    print(file)
    return {'Result': contents, 'executionTime': 'executionTime'}

    #import cv2
    #import numpy as np
    #decoded_image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
    #print(type(decoded_image))
    #from Recognize_face_image import compare_faces
    #result, execution = compare_faces(item_id, decoded_image)
    #execution = str(round(execution, 2))
    #print(result)
    #print(execution+'s')
    #return {'Result': result, 'executionTime': execution + 's'}


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
