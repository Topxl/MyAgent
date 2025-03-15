"""
Local audio transcription using Faster-Whisper, an optimized implementation of OpenAI's Whisper model.
"""
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import librosa
from pathlib import Path


class Transcription:
    """
    Audio transcription using Faster-Whisper, an optimized implementation of OpenAI's Whisper model.
    Converts audio signals to text with high accuracy, even in noisy environments.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu", 
                compute_type: str = "float16", language: Optional[str] = None):
        """
        Initialize the transcription module.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large").
            device: Device to run the model on ("cpu" or "cuda").
            compute_type: Compute type for the model ("float16", "float32", "int8").
            language: Language code for transcription (e.g., "en", "fr", None for auto-detection).
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        
        # Validate compute type based on device
        if self.device == "cpu" and self.compute_type == "float16":
            print("Warning: float16 not supported on CPU, switching to float32")
            self.compute_type = "float32"
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the Faster-Whisper model.
        """
        try:
            from faster_whisper import WhisperModel
            
            # Check if CUDA is available when device is set to cuda
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
                if self.compute_type == "float16":
                    self.compute_type = "float32"
            
            # Load the model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            print(f"Loaded Faster-Whisper model: {self.model_size}")
        except ImportError:
            print("Faster-Whisper not installed. Please install it with:")
            print("pip install faster-whisper")
            raise
    
    def transcribe(self, audio_path: str, segment: bool = True) -> Union[str, Dict]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file.
            segment: If True, return a dictionary with segments and timestamps.
                    If False, return just the text.
            
        Returns:
            Transcribed text or dictionary with detailed information.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please ensure Faster-Whisper is installed.")
        
        # Transcribe audio
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Process results
        if segment:
            result = {
                "text": "",
                "segments": [],
                "language": info.language,
                "language_probability": info.language_probability
            }
            
            # Process segments
            for segment in segments:
                result["text"] += segment.text + " "
                result["segments"].append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                              for word in segment.words] if segment.words else []
                })
            
            return result
        else:
            # Just return the concatenated text
            return " ".join(segment.text for segment in segments)
    
    def transcribe_large_file(self, audio_path: str, chunk_size_sec: int = 30,
                             overlap_sec: int = 5) -> Dict:
        """
        Transcribe a large audio file by processing it in chunks.
        
        Args:
            audio_path: Path to the audio file.
            chunk_size_sec: Size of each chunk in seconds.
            overlap_sec: Overlap between chunks in seconds.
            
        Returns:
            Dictionary with transcription results.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Calculate chunk size and overlap in samples
        chunk_size = chunk_size_sec * sr
        overlap = overlap_sec * sr
        
        # Process in chunks
        all_segments = []
        offset = 0
        
        while offset < len(audio):
            # Extract chunk
            end = min(offset + chunk_size, len(audio))
            chunk = audio[offset:end]
            
            # Save chunk to temporary file
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"chunk_{offset}.wav")
            librosa.output.write_wav(temp_file, chunk, sr)
            
            # Transcribe chunk
            chunk_result = self.transcribe(temp_file, segment=True)
            
            # Adjust timestamps
            chunk_start_time = offset / sr
            for segment in chunk_result["segments"]:
                segment["start"] += chunk_start_time
                segment["end"] += chunk_start_time
                all_segments.append(segment)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Move to next chunk with overlap
            offset = end - overlap
            if offset >= len(audio):
                break
        
        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        # Combine results
        result = {
            "text": " ".join(segment["text"] for segment in merged_segments),
            "segments": merged_segments,
            "language": chunk_result.get("language", "")
        }
        
        return result
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge overlapping segments in the transcription.
        
        Args:
            segments: List of segment dictionaries.
            
        Returns:
            List of merged segment dictionaries.
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            previous = merged[-1]
            
            # If segments overlap significantly
            if current["start"] < previous["end"] - 0.5:
                # Keep the longer segment or merge texts
                if current["end"] - current["start"] > previous["end"] - previous["start"]:
                    merged[-1] = current
                else:
                    # Optionally merge texts if they're different
                    if current["text"].strip() != previous["text"].strip():
                        merged[-1]["text"] += " " + current["text"]
                    
                    # Extend end time if needed
                    if current["end"] > previous["end"]:
                        merged[-1]["end"] = current["end"]
            else:
                merged.append(current)
        
        return merged
    
    def save_transcript(self, result: Dict, output_path: str, format: str = "txt"):
        """
        Save transcription results to a file.
        
        Args:
            result: Transcription result dictionary.
            output_path: Path to save the transcript.
            format: Output format ("txt", "srt", "vtt", "json").
        """
        if format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
        
        elif format == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"]):
                    # Format timestamp as SRT format
                    start_time = self._format_timestamp(segment["start"], srt=True)
                    end_time = self._format_timestamp(segment["end"], srt=True)
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
        
        elif format == "vtt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for i, segment in enumerate(result["segments"]):
                    # Format timestamp as VTT format
                    start_time = self._format_timestamp(segment["start"])
                    end_time = self._format_timestamp(segment["end"])
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
        
        elif format == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_timestamp(self, seconds: float, srt: bool = False) -> str:
        """
        Format seconds as a timestamp.
        
        Args:
            seconds: Time in seconds.
            srt: Whether to use SRT format (with comma) or VTT format (with period).
            
        Returns:
            Formatted timestamp string.
        """
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        
        if srt:
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        else:
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
