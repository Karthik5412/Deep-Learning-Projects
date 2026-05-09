import uvicorn
from fastapi import FastAPI
from funcs import preprocessing_data, tfidf_vectorize
import warnings
warnings.filterwarnings('ignore') 

app = FastAPI()
@app.get('/')
async def main_page():
    return 'All Good'

@app.get('/text_processing/')
async def text_processing(text : str) :
    matrix =  tfidf_vectorize([preprocessing_data(text)])
    
    return matrix.tolist()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

