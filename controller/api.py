from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from services.audio_service import AudioService

router = APIRouter()
audio_service = AudioService()

@router.post("/transcrever")
async def transcrever_audio_endpoint(
    file: UploadFile = File(None),
    url: str = Query(None, description="URL pública de áudio/vídeo"),
    model_name: str = Query("base", description="Nome do modelo Whisper (ex.: base, small, medium, large)"),
    remove_silencio: bool = Query(True, description="Se True, remove silêncio antes de transcrever")
):
    """
    Endpoint para transcrição de áudio/vídeo.
    """
    return await audio_service.transcrever_audio(
        file=file,
        url=url,
        model_name=model_name,
        remove_silencio=remove_silencio
    )

@router.post("/remover-silencio")
async def remover_silencio_endpoint(
    file: UploadFile = File(None),
    url: str = Query(None, description="URL pública de áudio/vídeo")
):
    """
    Endpoint para remoção de silêncio do áudio/vídeo.
    """

    processed_file, original_duration_ms, processed_duration_ms = await audio_service.remover_silencio(
        file=file,
        url=url
    )

    headers = {
        "X-Original-Duration-ms": str(original_duration_ms),
        "X-Processed-Duration-ms": str(processed_duration_ms),
    }

    return FileResponse(
        path=processed_file,
        media_type="audio/wav",
        filename="audio_sem_silencio.wav",
        headers=headers
    )
