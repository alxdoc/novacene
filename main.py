from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Assistant API", version="0.1.0")

class RequestModel(BaseModel):
    query: List[str] = Field(..., description="Query")

class ResponseModel(BaseModel):
    text: List[str] = Field(..., description="Text")
    links: List[str] = Field(..., description="Links")

class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationError]

@app.post("/assist", response_model=ResponseModel, responses={422: {"model": HTTPValidationError}}, tags=["default"])
async def assist(request: RequestModel):
    # Логика обработки запроса и генерации ответа и ссылок
    text_response = ["This is the response text based on the query."]
    links_response = ["https://link1.com", "https://link2.com"]
    return {"text": text_response, "links": links_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
