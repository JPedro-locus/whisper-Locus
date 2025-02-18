from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI()
model = whisper.load_model("base")  # ou outro tamanho/modelo desejado

@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    # Cria um arquivo temporário com delete=False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        temp_name = tmp.name  # Armazena o nome do arquivo

    try:
        # Agora o arquivo está fechado e o ffmpeg pode acessá-lo
        resultado = model.transcribe(temp_name)
    finally:
        # Remove o arquivo temporário, independentemente do resultado
        os.remove(temp_name)

    return {"transcricao": resultado["text"]}