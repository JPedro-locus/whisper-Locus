from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
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

def remove_silence(input_file_path: str, file_format: str):
    """
    Processa o áudio (ou vídeo) removendo trechos de silêncio.
    Retorna: (caminho_arquivo_WAV, duracao_original_ms, duracao_processada_ms)
    """
    audio = AudioSegment.from_file(input_file_path, format=file_format)

    original_duration = len(audio)  # ms

    chunks = split_on_silence(audio,
                              min_silence_len=100,    # mínimo de 100 ms de silêncio
                              silence_thresh=-35,     # limiar (dB)
                              keep_silence=50)        # manter 50 ms de silêncio no começo/fim de cada chunk

    processed_audio = sum(chunks)
    processed_duration = len(processed_audio)

    # Exporta para um arquivo WAV temporário
    temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    processed_audio.export(temp_output.name, format="wav")
    temp_output.close()

    return temp_output.name, original_duration, processed_duration


@app.post("/transcrever")
async def transcript_audio(
    file: UploadFile = File(...),
    model_name: str = Query("base", description="Nome do modelo Whisper (ex.: base, small, medium, large)"),
    remove_silencio: bool = Query(True, description="Se True, remove silêncio antes de transcrever")
):
    """
    Endpoint de transcrição (áudio ou vídeo).
    Se remove_silencio=True, aplica remoção de silêncio antes de transcrever.
    Retorna JSON com:
      - Transcrição
      - Duração original (em minutos)
      - Duração pós-processada (em minutos) [ou igual à original, se não removeu silêncio]
    """
    if file.filename and '.' in file.filename:
        original_extension = file.filename.rsplit('.', 1)[-1].lower()
    else:
        original_extension = 'wav'

    with tempfile.NamedTemporaryFile(suffix=f".{original_extension}", delete=False) as tmp:
        tmp.write(await file.read())
        original_temp_file = tmp.name

    processed_file = None
    original_duration_ms = None
    processed_duration_ms = None

    try:
        if remove_silencio:
            # Remove silêncio
            processed_file, original_duration_ms, processed_duration_ms = remove_silence(
                original_temp_file,
                file_format=original_extension
            )
        else:
            # Não remove silêncio - apenas trata "original_temp_file"
            # Mas ainda precisamos medir a duração do arquivo original
            audio_full = AudioSegment.from_file(original_temp_file, format=original_extension)
            original_duration_ms = len(audio_full)
            processed_duration_ms = original_duration_ms
            processed_file = original_temp_file

        # Carrega ou obtém modelo
        model = load_whisper_model(model_name)

        # Transcreve
        resultado = model.transcribe(processed_file)

    finally:
        # Se removeu silêncio, 'processed_file' é diferente de 'original_temp_file'.
        # Precisamos apagar ambos se for o caso.
        if os.path.exists(original_temp_file):
            os.remove(original_temp_file)
        if processed_file and os.path.exists(processed_file) and processed_file != original_temp_file:
            os.remove(processed_file)

    # Converte milissegundos para minutos
    original_duration_min = round(original_duration_ms / 1000.0 / 60.0, 2)
    processed_duration_min = round(processed_duration_ms / 1000.0 / 60.0, 2)

    return {
        "transcricao": resultado["text"],
        "duracao_original_min": original_duration_min,
        "duracao_processada_min": processed_duration_min
    }


@app.post("/remover-silencio")
async def remover_silencio_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para remover silêncio. Retorna o arquivo de áudio já sem silêncio.
    Se a entrada for vídeo, o pydub extrairá apenas o áudio, e a saída será WAV.
    """
    if file.filename and '.' in file.filename:
        original_extension = file.filename.rsplit('.', 1)[-1].lower()
    else:
        original_extension = 'wav'

    # Salva em arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=f".{original_extension}", delete=False) as tmp:
        tmp.write(await file.read())
        original_temp_file = tmp.name

    # Gera um arquivo WAV sem silêncio
    try:
        processed_file, original_duration, processed_duration = remove_silence(
            original_temp_file,
            file_format=original_extension
        )
    finally:
        if os.path.exists(original_temp_file):
            os.remove(original_temp_file)

    # Agora precisamos retornar esse WAV "limpo" como resposta
    # para que o usuário faça download (ou consuma) diretamente.
    # A forma mais simples é usar um FileResponse.
    # Se quiser, você poderia mudar o nome do arquivo final para algo mais amigável.
    file_name = "audio_sem_silencio.wav"

    # Exemplo de retorno de cabeçalhos para duração, se quiser:
    # (caso não precise, pode remover "headers")
    headers = {
        "X-Original-Duration-ms": str(original_duration),
        "X-Processed-Duration-ms": str(processed_duration),
    }

    return FileResponse(
        path=processed_file,
        media_type="audio/wav",
        filename=file_name,
        headers=headers
    )
