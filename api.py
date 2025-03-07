from fastapi import FastAPI
from parser import parse_proof
from translator import translate_to_coq

app = FastAPI()

@app.post("/translate/")
async def translate_proof(proof: str):
    parsed_statements = parse_proof(proof)
    coq_code = translate_to_coq(parsed_statements)
    return {"coq_code": coq_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
