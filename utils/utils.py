import tempfile
import requests
import boto3
from urllib.parse import urlparse
from fastapi import HTTPException
import whisper

def load_whisper_model(models: dict, model_name: str):
    """
    Carrega o modelo Whisper especificado.
    Se o modelo não estiver presente no dicionário 'models', realiza o carregamento.
    Retorna o modelo carregado.
    """
    if model_name not in models:
        models[model_name] = whisper.load_model(model_name)
    return models[model_name]

def download_file_from_s3(s3_url: str) -> (str, str):
    """
    Faz o download de um arquivo de um bucket S3 usando boto3.
    Retorna o caminho temporário e a extensão do arquivo.
    Suporta URLs do tipo:
      - s3://bucket/key
      - https://bucket.s3.amazonaws.com/key
    """
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme == "s3":
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
    elif parsed_url.scheme in ["http", "https"]:
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) > 2 and domain_parts[1] == "s3":
            bucket = domain_parts[0]
            key = parsed_url.path.lstrip('/')
        else:
            raise HTTPException(status_code=400, detail="URL S3 inválida.")
    else:
        raise HTTPException(status_code=400, detail="Esquema de URL inválido.")

    extension = key.split('.')[-1].lower() if '.' in key else "wav"

    # Inicializa o cliente S3 (certifique-se de que as credenciais AWS estejam configuradas)
    s3 = boto3.client("s3")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    try:
        s3.download_file(Bucket=bucket, Key=key, Filename=temp_file.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao baixar o arquivo S3: {e}")
    return temp_file.name, extension

def download_file(url: str) -> (str, str):
    """
    Baixa um arquivo de áudio ou vídeo a partir de uma URL e
    retorna o caminho temporário do arquivo e a extensão identificada.
    Se a URL indicar um objeto S3, utiliza o boto3.
    """
    if url.startswith("s3://") or "s3.amazonaws.com" in url:
        return download_file_from_s3(url)
    else:
        extension = url.split('.')[-1].lower()
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Erro ao baixar o arquivo.")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
        with open(temp_file.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_file.name, extension
