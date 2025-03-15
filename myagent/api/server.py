"""
API server for MyAgent using FastAPI.
This module provides a RESTful API for interacting with the MyAgent functionality.
"""
import os
import uuid
import json
import shutil
import base64
from typing import Dict, List, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import MyAgent components
from myagent.vad import VAD
from myagent.noise_reduction import NoiseReduction
from myagent.speaker_id import SpeakerID
from myagent.transcription import Transcription


# Models for API request/response
class TranscriptionResponse(BaseModel):
    id: str
    status: str
    transcription: Optional[str] = None
    speakers: Optional[List[Dict]] = None


class SpeakerResponse(BaseModel):
    speaker_id: str
    status: str


class ModelResponse(BaseModel):
    model_data: str
    speaker_id: str


# Initialize FastAPI app
app = FastAPI(
    title="MyAgent API",
    description="API for voice recording, speaker recognition, and transcription",
    version="0.1.0",
)

# Storage for background tasks
tasks = {}


def get_data_dir():
    """Get the data directory for storing uploaded files and results."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_temp_file(extension: str = ".wav") -> str:
    """Generate a temporary file path."""
    data_dir = get_data_dir()
    return os.path.join(data_dir, f"{uuid.uuid4()}{extension}")


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "MyAgent API",
        "version": "0.1.0",
        "description": "API for voice recording, speaker recognition, and transcription",
    }


@app.post("/upload", response_model=TranscriptionResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detect_speakers: bool = Form(True),
    language: Optional[str] = Form(None),
):
    """
    Upload an audio file for processing.
    
    The audio will be processed to detect speech, reduce noise, transcribe,
    and optionally identify speakers.
    """
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    data_dir = get_data_dir()
    task_dir = os.path.join(data_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_path = os.path.join(task_dir, "original" + os.path.splitext(file.filename)[1])
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Add task to background processing
    tasks[task_id] = {"status": "processing", "file_path": file_path}
    
    # Start processing in background
    background_tasks.add_task(
        process_audio, task_id, file_path, detect_speakers, language
    )
    
    return TranscriptionResponse(id=task_id, status="processing")


async def process_audio(
    task_id: str, file_path: str, detect_speakers: bool, language: Optional[str]
):
    """Process an audio file in the background."""
    try:
        # Initialize components
        vad = VAD()
        noise_reducer = NoiseReduction()
        transcriber = Transcription(language=language)
        
        # Process the audio
        # 1. Detect speech segments
        speech_segments = vad.detect(file_path)
        
        # 2. Reduce noise
        clean_audio_path = os.path.join(os.path.dirname(file_path), "clean.wav")
        clean_audio = noise_reducer.process(file_path, output_path=clean_audio_path)
        
        # 3. Transcribe
        transcription_result = transcriber.transcribe(clean_audio_path, segment=True)
        
        # 4. Identify speakers if requested
        speaker_segments = []
        if detect_speakers:
            speaker_id = SpeakerID()
            speaker_segments = speaker_id.identify_speakers_in_segments(
                clean_audio_path, transcription_result["segments"]
            )
        
        # Update task with results
        result = {
            "status": "completed",
            "transcription": transcription_result["text"],
            "segments": transcription_result["segments"],
            "speakers": speaker_segments if detect_speakers else []
        }
        
        tasks[task_id] = result
        
        # Save result to file
        result_path = os.path.join(os.path.dirname(file_path), "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f)
            
    except Exception as e:
        # Update task with error
        tasks[task_id] = {"status": "error", "error": str(e)}


@app.get("/transcription/{task_id}", response_model=TranscriptionResponse)
async def get_transcription(task_id: str):
    """
    Get the transcription and speaker information for a processed audio file.
    """
    if task_id not in tasks:
        # Check if result file exists
        data_dir = get_data_dir()
        task_dir = os.path.join(data_dir, task_id)
        result_path = os.path.join(task_dir, "result.json")
        
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            return TranscriptionResponse(
                id=task_id,
                status=result.get("status", "completed"),
                transcription=result.get("transcription", ""),
                speakers=result.get("speakers", [])
            )
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    
    return TranscriptionResponse(
        id=task_id,
        status=task.get("status", "unknown"),
        transcription=task.get("transcription", None),
        speakers=task.get("speakers", None)
    )


@app.post("/train_speaker", response_model=SpeakerResponse)
async def train_speaker(
    name: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Add a new speaker to the recognition model.
    """
    # Save the uploaded file
    file_path = get_temp_file(os.path.splitext(file.filename)[1])
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata format")
    
    # Add speaker to the model
    speaker_id = SpeakerID()
    speaker_id_str = speaker_id.add_speaker(file_path, name, metadata_dict)
    
    # Clean up temporary file
    os.unlink(file_path)
    
    return SpeakerResponse(speaker_id=speaker_id_str, status="trained")


@app.get("/export_model", response_model=ModelResponse)
async def export_model(speaker_id: str):
    """
    Export a trained speaker model.
    """
    # Get speaker recognition model
    speaker_id_model = SpeakerID()
    
    # Export the model
    model_path = get_temp_file(".pkl")
    success = speaker_id_model.export_model(model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to export model")
    
    # Read the model file and encode as base64
    with open(model_path, "rb") as f:
        model_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Clean up temporary file
    os.unlink(model_path)
    
    return ModelResponse(model_data=model_data, speaker_id=speaker_id)


@app.post("/import_model")
async def import_model(
    model_data: str = Form(...),
    merge: bool = Form(False)
):
    """
    Import a speaker model.
    """
    # Decode base64 model data
    try:
        model_bytes = base64.b64decode(model_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid model data: {str(e)}")
    
    # Save to temporary file
    model_path = get_temp_file(".pkl")
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    
    # Import the model
    speaker_id = SpeakerID()
    success = speaker_id.import_model(model_path, merge=merge)
    
    # Clean up temporary file
    os.unlink(model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to import model")
    
    return {"status": "success", "message": "Model imported successfully"}


def run(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the API server.
    
    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
