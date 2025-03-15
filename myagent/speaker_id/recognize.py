"""
Speaker identification module using SpeechBrain's ECAPA-TDNN model.
"""
import os
import torch
import torchaudio
import numpy as np
import pickle
import uuid
import datetime
from typing import Dict, List, Tuple, Optional, Union
import librosa
from speechbrain.pretrained import EncoderClassifier
from pathlib import Path
import json


class SpeakerID:
    """
    Speaker identification using SpeechBrain's ECAPA-TDNN model.
    Creates unique voice fingerprints for each speaker and can identify speakers in recordings.
    """
    
    def __init__(self, model_dir: Optional[str] = None, similarity_threshold: float = 0.75):
        """
        Initialize the speaker identification system.
        
        Args:
            model_dir: Directory to store speaker embeddings and models.
            similarity_threshold: Threshold for speaker similarity (0.0 to 1.0).
        """
        # Set up model directory
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "speaker_id")
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Speaker database path
        self.speaker_db_path = os.path.join(self.model_dir, "speaker_db.pkl")
        self.speaker_info_path = os.path.join(self.model_dir, "speaker_info.json")
        
        # Initialize speaker database
        self.speaker_db = self._load_speaker_db()
        self.speaker_info = self._load_speaker_info()
        
        # Similarity threshold for speaker identification
        self.similarity_threshold = similarity_threshold
        
        # Load SpeechBrain's ECAPA-TDNN model
        self._load_embedder()
    
    def _load_embedder(self):
        """
        Load the SpeechBrain ECAPA-TDNN model for speaker embedding extraction.
        """
        try:
            self.embedder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(self.model_dir, "ecapa_model")
            )
            print("Loaded SpeechBrain ECAPA-TDNN model")
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            print("Installing SpeechBrain...")
            raise ImportError("SpeechBrain not installed. Please install it with: pip install speechbrain")
    
    def _load_speaker_db(self) -> Dict[str, np.ndarray]:
        """
        Load the speaker database from disk.
        
        Returns:
            Dictionary of speaker IDs to speaker embeddings.
        """
        if os.path.exists(self.speaker_db_path):
            try:
                with open(self.speaker_db_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading speaker database: {e}")
                return {}
        else:
            return {}
    
    def _save_speaker_db(self):
        """
        Save the speaker database to disk.
        """
        try:
            with open(self.speaker_db_path, 'wb') as f:
                pickle.dump(self.speaker_db, f)
        except Exception as e:
            print(f"Error saving speaker database: {e}")
    
    def _load_speaker_info(self) -> Dict[str, Dict]:
        """
        Load the speaker information from disk.
        
        Returns:
            Dictionary of speaker IDs to speaker information.
        """
        if os.path.exists(self.speaker_info_path):
            try:
                with open(self.speaker_info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading speaker information: {e}")
                return {}
        else:
            return {}
    
    def _save_speaker_info(self):
        """
        Save the speaker information to disk.
        """
        try:
            with open(self.speaker_info_path, 'w') as f:
                json.dump(self.speaker_info, f, indent=2)
        except Exception as e:
            print(f"Error saving speaker information: {e}")
    
    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Speaker embedding as numpy array.
        """
        # Load and preprocess audio
        signal, fs = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Extract embedding using SpeechBrain
        embeddings = self.embedder.encode_batch(signal)
        embedding = embeddings.squeeze().detach().cpu().numpy()
        
        return embedding
    
    def add_speaker(self, audio_path: str, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a new speaker to the database.
        
        Args:
            audio_path: Path to an audio file of the speaker.
            name: Name of the speaker.
            metadata: Optional metadata for the speaker.
            
        Returns:
            Speaker ID.
        """
        # Generate a unique ID for the speaker
        speaker_id = str(uuid.uuid4())
        
        # Extract speaker embedding
        embedding = self._extract_embedding(audio_path)
        
        # Add to database
        self.speaker_db[speaker_id] = embedding
        
        # Add speaker info
        if metadata is None:
            metadata = {}
        
        self.speaker_info[speaker_id] = {
            "name": name,
            "metadata": metadata,
            "created_at": str(datetime.datetime.now())
        }
        
        # Save to disk
        self._save_speaker_db()
        self._save_speaker_info()
        
        print(f"Added speaker '{name}' with ID {speaker_id}")
        return speaker_id
    
    def identify_speaker(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Identify the speaker in an audio file.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (speaker_id, confidence) or (None, 0.0) if no match.
        """
        if not self.speaker_db:
            return None, 0.0
        
        # Extract embedding from the audio
        embedding = self._extract_embedding(audio_path)
        
        # Compare with all speakers in the database
        best_match = None
        best_score = 0.0
        
        for speaker_id, speaker_embedding in self.speaker_db.items():
            # Calculate cosine similarity
            similarity = self._calculate_similarity(embedding, speaker_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = speaker_id
        
        # Only return a match if it's above the threshold
        if best_score >= self.similarity_threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def identify_speakers_in_segments(self, audio_path: str, segments: List[Dict[str, float]]) -> List[Dict]:
        """
        Identify speakers in each segment of an audio file.
        
        Args:
            audio_path: Path to the audio file.
            segments: List of dictionaries containing 'start' and 'end' times in seconds.
            
        Returns:
            List of dictionaries with segment information and identified speakers.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        results = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Save segment to temporary file
            temp_file = os.path.join(self.model_dir, "temp_segment.wav")
            librosa.output.write_wav(temp_file, segment_audio, sr)
            
            # Identify speaker
            speaker_id, confidence = self.identify_speaker(temp_file)
            
            # Get speaker name if available
            speaker_name = None
            if speaker_id and speaker_id in self.speaker_info:
                speaker_name = self.speaker_info[speaker_id]["name"]
            
            # Add to results
            results.append({
                "start": segment['start'],
                "end": segment['end'],
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "confidence": confidence
            })
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return results
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete a speaker from the database.
        
        Args:
            speaker_id: ID of the speaker to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        if speaker_id in self.speaker_db:
            del self.speaker_db[speaker_id]
            
            if speaker_id in self.speaker_info:
                del self.speaker_info[speaker_id]
            
            self._save_speaker_db()
            self._save_speaker_info()
            
            return True
        
        return False
    
    def update_speaker(self, speaker_id: str, name: Optional[str] = None, 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Update a speaker's information.
        
        Args:
            speaker_id: ID of the speaker to update.
            name: New name for the speaker (if None, keep existing).
            metadata: New metadata for the speaker (if None, keep existing).
            
        Returns:
            True if successful, False otherwise.
        """
        if speaker_id in self.speaker_info:
            if name:
                self.speaker_info[speaker_id]["name"] = name
            
            if metadata:
                self.speaker_info[speaker_id]["metadata"] = metadata
            
            self._save_speaker_info()
            return True
        
        return False
    
    def export_model(self, output_path: str) -> bool:
        """
        Export the speaker model to a file for sharing.
        
        Args:
            output_path: Path to save the exported model.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {
                "speaker_db": self.speaker_db,
                "speaker_info": self.speaker_info,
                "threshold": self.similarity_threshold
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            print(f"Error exporting model: {e}")
            return False
    
    def import_model(self, model_path: str, merge: bool = False) -> bool:
        """
        Import a speaker model from a file.
        
        Args:
            model_path: Path to the model file.
            merge: If True, merge with existing model. If False, replace existing model.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            if not merge:
                self.speaker_db = data["speaker_db"]
                self.speaker_info = data["speaker_info"]
            else:
                # Merge with existing model
                self.speaker_db.update(data["speaker_db"])
                self.speaker_info.update(data["speaker_info"])
            
            # Set threshold if provided
            if "threshold" in data:
                self.similarity_threshold = data["threshold"]
            
            self._save_speaker_db()
            self._save_speaker_info()
            
            return True
        except Exception as e:
            print(f"Error importing model: {e}")
            return False
    
    def reinforce_speaker(self, speaker_id: str, audio_path: str, weight: float = 0.3) -> bool:
        """
        Reinforce a speaker model with new audio data.
        
        Args:
            speaker_id: ID of the speaker to reinforce.
            audio_path: Path to the new audio file.
            weight: Weight of the new embedding (0.0 to 1.0).
            
        Returns:
            True if successful, False otherwise.
        """
        if speaker_id not in self.speaker_db:
            return False
        
        # Extract new embedding
        new_embedding = self._extract_embedding(audio_path)
        
        # Get existing embedding
        existing_embedding = self.speaker_db[speaker_id]
        
        # Weighted average
        updated_embedding = (1 - weight) * existing_embedding + weight * new_embedding
        
        # Normalize
        updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
        
        # Update database
        self.speaker_db[speaker_id] = updated_embedding
        self._save_speaker_db()
        
        return True
    
    def list_speakers(self) -> List[Dict]:
        """
        List all speakers in the database.
        
        Returns:
            List of dictionaries with speaker information.
        """
        result = []
        
        for speaker_id, info in self.speaker_info.items():
            result.append({
                "id": speaker_id,
                "name": info["name"],
                "metadata": info.get("metadata", {})
            })
        
        return result
