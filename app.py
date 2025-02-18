from fastapi import FastAPI, UploadFile, File, Query
import whisper
import tempfile
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

app = FastAPI()

# Dicionário global para armazenar os modelos carregados
models = {}

def load_whisper_model(model_name: str):
    """Carrega o modelo Whisper especificado, se ainda não estiver carregado."""
    if model_name not in models:
        models[model_name] = whisper.load_model(model_name)
    return models[model_name]

def remove_silence(input_file_path: str, file_format: str) -> str:
    """
    Processa o áudio removendo trechos de silêncio e salva em um novo arquivo.
    Retorna o caminho para o arquivo processado.
    """
    # Carrega o áudio usando o formato correto
    audio = AudioSegment.from_file(input_file_path, format=file_format)
    # Separa os trechos sem silêncio (ajuste os parâmetros conforme necessário)
    chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-35, keep_silence=50)
    # Junta os trechos (a lógica aqui pode ser customizada)
    processed_audio = sum(chunks)
    # Cria um arquivo temporário para salvar o áudio processado (aqui exportamos para WAV)
    temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    processed_audio.export(temp_output.name, format="wav")
    temp_output.close()  # Garante que o arquivo esteja fechado para acesso por outros processos
    return temp_output.name

@app.post("/transcrever")
async def transcrever_audio(
    file: UploadFile = File(...),
    model_name: str = Query("base", description="Nome do modelo Whisper a ser usado (ex.: base, small, medium, large)")
):
    # Detecta a extensão do arquivo enviado (ex.: mp3, wav, etc.)
    if file.filename and '.' in file.filename:
        original_extension = file.filename.rsplit('.', 1)[-1].lower()
    else:
        original_extension = 'wav'
    
    # Salva o arquivo enviado em um arquivo temporário usando a extensão correta
    with tempfile.NamedTemporaryFile(suffix=f".{original_extension}", delete=False) as tmp:
        tmp.write(await file.read())
        original_temp_file = tmp.name

    processed_file = None
    try:
        # Processa o áudio para remover silêncios, usando o formato detectado
        processed_file = remove_silence(original_temp_file, file_format=original_extension)
        # Carrega o modelo especificado (ou usa um já carregado)
        model = load_whisper_model(model_name)
        # Realiza a transcrição utilizando o modelo selecionado e o áudio processado
        resultado = model.transcribe(processed_file)
    finally:
        # Remove os arquivos temporários criados
        if os.path.exists(original_temp_file):
            os.remove(original_temp_file)
        if processed_file and os.path.exists(processed_file):
            os.remove(processed_file)

    return {"transcricao": resultado["text"]}
