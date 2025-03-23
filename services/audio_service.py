from fastapi import UploadFile, HTTPException
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import tempfile
from utils.utils import load_whisper_model, download_file 
from config import SILENCE_MIN_LENGTH, SILENCE_THRESH, KEEP_SILENCE

class AudioService:
    def __init__(self):
        self._models = {}

    async def transcrever_audio(self, file: UploadFile, url: str, model_name: str, remove_silencio: bool) -> dict:

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
            original_temp_file, original_extension = download_file(url)

        try:
            audio_full = AudioSegment.from_file(original_temp_file, format=original_extension)
            original_duration_ms = len(audio_full)
            if remove_silencio:
                chunks = split_on_silence(
                    audio_full,
                    min_silence_len=SILENCE_MIN_LENGTH, 
                    silence_thresh=SILENCE_THRESH,       
                    keep_silence=KEEP_SILENCE  
                )
                processed_audio = sum(chunks)
                processed_duration_ms = len(processed_audio)
                temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                processed_audio.export(temp_output.name, format="wav")
                temp_output.close()
                processed_file = temp_output.name
            else:
                processed_file = original_temp_file
                processed_duration_ms = original_duration_ms

            model = load_whisper_model(self._models, model_name)
            resultado = model.transcribe(processed_file)
        finally:
            try:
                if os.path.exists(original_temp_file):
                    os.remove(original_temp_file)
            except Exception as e:
                print(f"Erro ao remover arquivo temporário {original_temp_file}: {e}")
            if remove_silencio and 'processed_file' in locals() and processed_file != original_temp_file:
                try:
                    if os.path.exists(processed_file):
                        os.remove(processed_file)
                except Exception as e:
                    print(f"Erro ao remover arquivo temporário {processed_file}: {e}")

        original_duration_min = round(original_duration_ms / 1000.0 / 60.0, 2)
        processed_duration_min = round(processed_duration_ms / 1000.0 / 60.0, 2)

        return {
            "transcricao": resultado["text"],
            "duracao_original_min": original_duration_min,
            "duracao_processada_min": processed_duration_min
        }
    
    async def remover_silencio(self, file: UploadFile, url: str) -> (str, int, int):
        """
        Remove o silêncio de um áudio/vídeo e retorna o caminho do arquivo processado,
        juntamente com as durações original e processada (em milissegundos).
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
            original_temp_file, original_extension = download_file(url)

        try:
            audio = AudioSegment.from_file(original_temp_file, format=original_extension)
            original_duration_ms = len(audio)

            chunks = split_on_silence(
                audio,
                min_silence_len=SILENCE_MIN_LENGTH,  
                silence_thresh=SILENCE_THRESH,   
                keep_silence=KEEP_SILENCE 
            )
            processed_audio = sum(chunks)
            processed_duration_ms = len(processed_audio)

            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            processed_audio.export(temp_output.name, format="wav")
            temp_output.close()
            processed_file = temp_output.name

        finally:
            try:
                if os.path.exists(original_temp_file):
                    os.remove(original_temp_file)
            except Exception as e:
                print(f"Erro ao remover arquivo temporário {original_temp_file}: {e}")

        return processed_file, original_duration_ms, processed_duration_ms
