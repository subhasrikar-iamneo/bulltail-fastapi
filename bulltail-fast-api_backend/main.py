from fastapi import FastAPI , File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import emodet
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import shutil
import agepre
import twilioo

# Configuration       
cloudinary.config( 
    cloud_name = "dgfez6arq", 
    api_key = "682424119974336", 
    api_secret = "jCFMOgCbgujHztXsYJW3r-UiiRk", # Click 'View API Keys' above to copy your API secret
    secure=True
)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class Task(BaseModel):
    user: str
    password: str
    
class Message(BaseModel):
    text: str

tasks= []

adminpass={ "user":"srikarsubha@gmail.com",
           "password":"yaahoo"
           }
tasks.append(Task(user="srikarsubha@gmail.com", password="yaahoo"))

print(tasks[0].user)

@app.get("/validation/{user}/{password}")
def is_valid(user: str, password: str) -> dict:
    for task in tasks:
        if task.user == user:
            if task.password == password:
                return {"is_valid": True, "message": "Validation successful"}
            else:
                return {"is_valid": False, "message": "Invalid password"}
    return {"is_valid": False, "message": "User not found"}

@app.get("/emodet/{sentence}")
def emodetc(sentence: str):
    pred=emodet.predic(sentence)
    return {"emotion": pred}

@app.get("/emolime/{sentence}")
def emolime(sentence: str):
    pred=emodet.explain(sentence)
    # Upload an image
    upload_result = cloudinary.uploader.upload("lime.png",public_id="lime")
    predi=upload_result["secure_url"]
    return {"emotion": predi}

@app.get("/imgpred")
def pim():
    gender,age=agepre.predicage()
    return {"gender":gender,"age":age}

@app.post("/post-msg/")
async def post_string(message: Message):
    twilioo.sendmsg(message.text)
    return {"message": f"You sent: {message.text}"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return success response
        return JSONResponse(content={"message": "Image uploaded successfully!", "filename": file.filename}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"message": f"Error uploading image: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)