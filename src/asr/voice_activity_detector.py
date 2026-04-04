"""Voice Activity Detection module for detecting speech in audio input.

Implements noise reduction and VAD algorithms to ensure robust performance
in classroom and real-world environments as described in the paper.
"""

import numpy as np


class VoiceActivityDetector:
    """Detects voice activity in audio streams using energy-based and
    zero-crossing rate methods for efficient activation of the ASR pipeline."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 frame_duration_ms: int = 30, min_speech_duration_ms: int = 300):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self._noise_floor = None
        self._calibration_frames = 0
        self._calibration_target = 10

    def _compute_energy(self, frame: np.ndarray) -> float:
        """Compute short-time energy of an audio frame."""
        return float(np.sum(frame.astype(np.float64) ** 2) / len(frame))

    def _compute_zcr(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate of an audio frame."""
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / len(frame))

    def _compute_spectral_flatness(self, frame: np.ndarray) -> float:
        """Compute spectral flatness to distinguish speech from noise."""
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum[spectrum > 0]
        if len(spectrum) == 0:
            return 0.0
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arithmetic_mean = np.mean(spectrum)
        if arithmetic_mean == 0:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    def calibrate_noise_floor(self, audio: np.ndarray):
        """Calibrate noise floor from ambient audio for adaptive thresholding."""
        frames = [audio[i:i + self.frame_size]
                  for i in range(0, len(audio) - self.frame_size, self.frame_size)]
        energies = [self._compute_energy(f) for f in frames]
        self._noise_floor = np.mean(energies) if energies else 0.0

    def detect(self, audio: np.ndarray) -> dict:
        """Detect voice activity in audio signal.

        Returns:
            Dictionary with 'is_speech', 'confidence', 'speech_segments'
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        frames = [audio[i:i + self.frame_size]
                  for i in range(0, len(audio) - self.frame_size, self.frame_size)]

        if not frames:
            return {"is_speech": False, "confidence": 0.0, "speech_segments": []}

        # Auto-calibrate noise floor if not set
        if self._noise_floor is None:
            self._noise_floor = self._compute_energy(frames[0]) if frames else 0.001

        frame_decisions = []
        frame_confidences = []

        for frame in frames:
            energy = self._compute_energy(frame)
            zcr = self._compute_zcr(frame)
            flatness = self._compute_spectral_flatness(frame)

            # Energy ratio above noise floor
            energy_ratio = energy / (self._noise_floor + 1e-10)
            energy_score = min(1.0, energy_ratio / 10.0)

            # ZCR score (speech typically has moderate ZCR)
            zcr_score = 1.0 - abs(zcr - 0.1) / 0.3
            zcr_score = max(0.0, min(1.0, zcr_score))

            # Spectral flatness (noise is flatter than speech)
            flatness_score = 1.0 - flatness

            # Combined confidence
            confidence = 0.5 * energy_score + 0.3 * zcr_score + 0.2 * flatness_score
            is_speech = confidence >= self.threshold

            frame_decisions.append(is_speech)
            frame_confidences.append(confidence)

        # Find contiguous speech segments
        speech_segments = []
        segment_start = None
        consecutive_speech = 0

        for i, is_speech in enumerate(frame_decisions):
            if is_speech:
                if segment_start is None:
                    segment_start = i
                consecutive_speech += 1
            else:
                if segment_start is not None and consecutive_speech >= self.min_speech_frames:
                    start_sample = segment_start * self.frame_size
                    end_sample = i * self.frame_size
                    speech_segments.append({
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "start_time": start_sample / self.sample_rate,
                        "end_time": end_sample / self.sample_rate,
                        "confidence": float(np.mean(frame_confidences[segment_start:i])),
                    })
                segment_start = None
                consecutive_speech = 0

        # Handle segment at end of audio
        if segment_start is not None and consecutive_speech >= self.min_speech_frames:
            start_sample = segment_start * self.frame_size
            end_sample = len(audio)
            speech_segments.append({
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": start_sample / self.sample_rate,
                "end_time": end_sample / self.sample_rate,
                "confidence": float(np.mean(frame_confidences[segment_start:])),
            })

        overall_speech = len(speech_segments) > 0
        overall_confidence = float(np.mean(frame_confidences)) if frame_confidences else 0.0

        return {
            "is_speech": overall_speech,
            "confidence": overall_confidence,
            "speech_segments": speech_segments,
        }

    def extract_speech(self, audio: np.ndarray) -> np.ndarray | None:
        """Extract speech portions from audio, removing silence/noise."""
        result = self.detect(audio)
        if not result["is_speech"]:
            return None

        speech_parts = []
        for seg in result["speech_segments"]:
            speech_parts.append(audio[seg["start_sample"]:seg["end_sample"]])

        if speech_parts:
            return np.concatenate(speech_parts)
        return None
