from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import whisper
import tempfile
import os
import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence

app = FastAPI()

class AudioService:
    """
    Classe 'pública' responsável pelos métodos de processamento de áudio.
    Métodos com underscore são considerados 'privados' por convenção.
    """

    def __init__(self):
        self._models = {}

    def _load_whisper_model(self, model_name: str):
        """
        Método 'privado' (por convenção) que carrega o modelo Whisper especificado.
        """
        if model_name not in self._models:
            self._models[model_name] = whisper.load_model(model_name)
        return self._models[model_name]

    def _download_file(self, url: str) -> str:
        """
        Método 'privado' (por convenção) que baixa um arquivo de áudio ou vídeo a partir de uma URL pública.
        Retorna o caminho temporário no sistema de arquivos.
        """
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Erro ao baixar o arquivo.")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_file.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_file.name

    async def transcrever_audio(
        self, 
        file: UploadFile,
        url: str,
        model_name: str,
        remove_silencio: bool
    ) -> dict:
        """
        Método público de transcrição de áudio (ou vídeo).
        Se 'remove_silencio=True', aplica a remoção de silêncio antes de transcrever.
        Retorna um dicionário com a transcrição e as durações (original e processada).
        """
        if not file and not url:
            raise HTTPException(status_code=400, detail="Envie um arquivo ou uma URL de áudio/vídeo.")
        if file and url:
            raise HTTPException(status_code=400, detail="Envie apenas um tipo de entrada (arquivo OU URL).")

        # Salva em arquivo temporário ou baixa via URL
        if file:
            # Se for UploadFile
            if file.filename and '.' in file.filename:
                original_extension = file.filename.rsplit('.', 1)[-1].lower()
            else:
                original_extension = "wav"

            with tempfile.NamedTemporaryFile(suffix=f".{original_extension}", delete=False) as tmp:
                tmp.write(await file.read())
                original_temp_file = tmp.name
        else:
            # Se for URL
            original_temp_file = self._download_file(url)
            original_extension = "wav"

        try:
            audio_full = AudioSegment.from_file(original_temp_file, format=original_extension)
            original_duration_ms = len(audio_full)

            if remove_silencio:
                chunks = split_on_silence(
                    audio_full,
                    min_silence_len=100,   # 100 ms de silêncio
                    silence_thresh=-35,    # limiar (dB)
                    keep_silence=50        # mantém 50 ms de silêncio no início/fim de cada chunk
                )
                processed_audio = sum(chunks)
                processed_duration_ms = len(processed_audio)

                # Exporta para WAV
                temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                processed_audio.export(temp_output.name, format="wav")
                temp_output.close()
                processed_file = temp_output.name
            else:
                processed_file = original_temp_file
                processed_duration_ms = original_duration_ms

            # Carrega (ou obtém) o modelo
            model = self._load_whisper_model(model_name)
            # Transcreve
            resultado = model.transcribe(processed_file)

        finally:
            # Remove arquivos temporários
            if os.path.exists(original_temp_file):
                os.remove(original_temp_file)
            if remove_silencio and processed_file != original_temp_file and os.path.exists(processed_file):
                os.remove(processed_file)

        # Converte ms para minutos
        original_duration_min = round(original_duration_ms / 1000.0 / 60.0, 2)
        processed_duration_min = round(processed_duration_ms / 1000.0 / 60.0, 2)

        return {
            "transcricao": resultado["text"],
            "duracao_original_min": original_duration_min,
            "duracao_processada_min": processed_duration_min
        }

    async def remover_silencio(
        self,
        file: UploadFile,
        url: str
    ) -> FileResponse:
        """
        Método público para remover silêncio de um áudio/vídeo e retornar o arquivo WAV resultante.
        """
        if not file and not url:
            raise HTTPException(status_code=400, detail="Envie um arquivo ou uma URL de áudio/vídeo.")
        if file and url:
            raise HTTPException(status_code=400, detail="Envie apenas um tipo de entrada (arquivo OU URL).")

        if file:
            if file.filename and '.' in file.filename:
                original_extension = file.filename.rsplit('.', 1)[-1].lower()
            else:
                original_extension = "wav"

            with tempfile.NamedTemporaryFile(suffix=f".{original_extension}", delete=False) as tmp:
                tmp.write(await file.read())
                original_temp_file = tmp.name
        else:
            original_temp_file = self._download_file(url)
            original_extension = "wav"

        try:
            audio = AudioSegment.from_file(original_temp_file, format=original_extension)
            original_duration_ms = len(audio)

            chunks = split_on_silence(
                audio,
                min_silence_len=100,
                silence_thresh=-35,
                keep_silence=50
            )
            processed_audio = sum(chunks)
            processed_duration_ms = len(processed_audio)

            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            processed_audio.export(temp_output.name, format="wav")
            temp_output.close()
            processed_file = temp_output.name

        finally:
            if os.path.exists(original_temp_file):
                os.remove(original_temp_file)

        file_name = "audio_sem_silencio.wav"
        headers = {
            "X-Original-Duration-ms": str(original_duration_ms),
            "X-Processed-Duration-ms": str(processed_duration_ms),
        }

        return FileResponse(
            path=processed_file,
            media_type="audio/wav",
            filename=file_name,
            headers=headers
        )

audio_service = AudioService()

@app.post("/transcrever")
async def transcrever_audio_endpoint(
    file: UploadFile = File(None),
    url: str = Query(None, description="URL pública de áudio/vídeo"),
    model_name: str = Query("base", description="Nome do modelo Whisper (ex.: base, small, medium, large)"),
    remove_silencio: bool = Query(True, description="Se True, remove silêncio antes de transcrever")
):
    """
    Endpoint público que chama o método transcrever_audio da nossa classe.
    """
    return await audio_service.transcrever_audio(
        file=file,
        url=url,
        model_name=model_name,
        remove_silencio=remove_silencio
    )

@app.post("/remover-silencio")
async def remover_silencio_endpoint(
    file: UploadFile = File(None),
    url: str = Query(None, description="URL pública de áudio/vídeo")
):
    """
    Endpoint público que chama o método remover_silencio da nossa classe.
    """
    return await audio_service.remover_silencio(
        file=file,
        url=url
    )
