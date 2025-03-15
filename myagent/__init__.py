"""
MyAgent - Advanced Open-Source Voice Analysis & Transcription Framework
"""

__version__ = "0.1.0"

from myagent.vad import VAD
from myagent.noise_reduction import NoiseReduction
from myagent.speaker_id import SpeakerID
from myagent.transcription import Transcription

__all__ = ["VAD", "NoiseReduction", "SpeakerID", "Transcription"]
