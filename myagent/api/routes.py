"""
API routes for MyAgent.
This module defines the FastAPI routes for the MyAgent API.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import os
import uuid
import json
import shutil
from pathlib import Path

# Import MyAgent components
from myagent.vad import VAD
from myagent.noise_reduction import NoiseReduction
from myagent.speaker_id import SpeakerID
from myagent.transcription import Transcription
from myagent.api.server import get_data_dir, tasks, process_audio

# Create router
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "MyAgent API",
        "version": "0.1.0",
        "description": "API for voice recording, speaker recognition, and transcription",
    }


@router.post("/upload")
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
    
    return {"id": task_id, "status": "processing"}


@router.get("/transcription/{task_id}")
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
            return {
                "id": task_id,
                "status": result.get("status", "completed"),
                "transcription": result.get("transcription", ""),
                "speakers": result.get("speakers", [])
            }
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    
    return {
        "id": task_id,
        "status": task.get("status", "unknown"),
        "transcription": task.get("transcription", None),
        "speakers": task.get("speakers", None)
    }


@router.post("/train_speaker")
async def train_speaker(
    name: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Add a new speaker to the recognition model.
    """
    # Save the uploaded file to a temporary location
    data_dir = get_data_dir()
    temp_dir = os.path.join(data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
    
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
    
    return {"speaker_id": speaker_id_str, "status": "trained"}


@router.get("/export_model")
async def export_model(speaker_id: str):
    """
    Export a trained speaker model.
    """
    import base64
    
    # Get speaker recognition model
    speaker_id_model = SpeakerID()
    
    # Export the model
    data_dir = get_data_dir()
    temp_dir = os.path.join(data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    model_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pkl")
    success = speaker_id_model.export_model(model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to export model")
    
    # Read the model file and encode as base64
    with open(model_path, "rb") as f:
        model_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Clean up temporary file
    os.unlink(model_path)
    
    return {"model_data": model_data, "speaker_id": speaker_id}


@router.post("/import_model")
async def import_model(
    model_data: str = Form(...),
    merge: bool = Form(False)
):
    """
    Import a speaker model.
    """
    import base64
    
    # Decode base64 model data
    try:
        model_bytes = base64.b64decode(model_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid model data: {str(e)}")
    
    # Save to temporary file
    data_dir = get_data_dir()
    temp_dir = os.path.join(data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    model_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pkl")
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
