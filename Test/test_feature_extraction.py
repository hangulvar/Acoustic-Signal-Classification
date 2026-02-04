"""
Unit tests for feature extraction optimizations.
Tests new functions: pre-emphasis, chunk validation, and optimized parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from Feature_Extraction import (
    normalize_rms, chunk_audio, extract_spectrogram,
    apply_preemphasis, is_valid_chunk,
    TARGET_SR, CHUNK_SEC, N_MELS, N_FFT, HOP_LENGTH
)

class TestNormalization:
    """Test RMS normalization function."""
    
    def test_normalize_rms_increases_amplitude(self):
        """Test that quiet audio is amplified."""
        audio = np.random.randn(1000) * 0.01  # Very quiet
        normalized = normalize_rms(audio, target_rms=0.1)
        
        # Check that RMS is close to target
        actual_rms = np.sqrt(np.mean(normalized**2))
        assert abs(actual_rms - 0.1) < 0.01
        
    def test_normalize_rms_decreases_amplitude(self):
        """Test that loud audio is attenuated."""
        audio = np.random.randn(1000) * 1.0  # Loud
        normalized = normalize_rms(audio, target_rms=0.1)
        
        actual_rms = np.sqrt(np.mean(normalized**2))
        assert abs(actual_rms - 0.1) < 0.01
        
    def test_normalize_rms_handles_silence(self):
        """Test that silence doesn't cause division by zero."""
        audio = np.zeros(1000)
        normalized = normalize_rms(audio, target_rms=0.1)
        
        # Should return zeros without crashing
        assert np.allclose(normalized, 0.0)


class TestChunking:
    """Test audio chunking function."""
    
    def test_chunk_audio_creates_correct_chunks(self):
        """Test that chunks have correct length."""
        sr = 22050
        chunk_sec = 5
        overlap_sec = 2.5
        
        # Create 10-second audio
        audio = np.random.randn(10 * sr)
        chunks = chunk_audio(audio, sr, chunk_sec, overlap_sec)
        
        # Each chunk should be 5 seconds
        expected_length = int(chunk_sec * sr)
        for chunk in chunks:
            assert len(chunk) == expected_length
            
    def test_chunk_audio_pads_short_audio(self):
        """Test that short audio is padded."""
        sr = 22050
        chunk_sec = 5
        overlap_sec = 2.5
        
        # Create 2-second audio (shorter than chunk)
        audio = np.random.randn(2 * sr)
        chunks = chunk_audio(audio, sr, chunk_sec, overlap_sec)
        
        # Should return one padded chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == int(chunk_sec * sr)


class TestSpectrogramExtraction:
    """Test spectrogram extraction function."""
    
    def test_extract_spectrogram_shape(self):
        """Test that spectrogram has correct shape."""
        sr = 22050
        chunk_sec = 5
        audio = np.random.randn(int(chunk_sec * sr))
        
        spec = extract_spectrogram(audio, sr)
        
        # Should have N_MELS frequency bins
        assert spec.shape[0] == N_MELS
        
        # Time dimension should be approximately correct
        assert spec.shape[1] > 0
        
        print(f"  ℹ️  Spectrogram shape: {spec.shape} (N_MELS={N_MELS}, N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH})")
        
    def test_extract_spectrogram_empty_audio(self):
        """Test that empty audio raises error."""
        with pytest.raises(ValueError, match="Empty audio chunk"):
            extract_spectrogram(np.array([]), 22050)
            
    def test_extract_spectrogram_invalid_sr(self):
        """Test that invalid sample rate raises error."""
        audio = np.random.randn(1000)
        with pytest.raises(ValueError, match="Invalid sample rate"):
            extract_spectrogram(audio, -1)
            
    def test_extract_spectrogram_return_shape(self):
        """Test return_shape parameter."""
        sr = 22050
        audio = np.random.randn(int(5 * sr))
        
        spec, shape = extract_spectrogram(audio, sr, return_shape=True)
        
        assert shape == spec.shape
        assert spec.shape[0] == N_MELS


class TestPreemphasis:
    """Test pre-emphasis filter."""
    
    def test_preemphasis_shape(self):
        """Test that pre-emphasis preserves audio length."""
        audio = np.random.randn(1000)
        filtered = apply_preemphasis(audio, coef=0.97)
        
        assert len(filtered) == len(audio)
        
    def test_preemphasis_highpass_effect(self):
        """Test that pre-emphasis acts as high-pass filter."""
        # Create a simple signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Low frequency component should be attenuated more than high frequency
        low_freq = np.sin(2 * np.pi * 100 * t)  # 100 Hz
        high_freq = np.sin(2 * np.pi * 5000 * t)  # 5000 Hz
        
        filtered_low = apply_preemphasis(low_freq, coef=0.97)
        filtered_high = apply_preemphasis(high_freq, coef=0.97)
        
        # High frequency should have higher RMS after filtering (relatively boosted)
        rms_low = np.sqrt(np.mean(filtered_low**2))
        rms_high = np.sqrt(np.mean(filtered_high**2))
        
        # The high frequency should have higher energy
        assert rms_high > rms_low
        
    def test_preemphasis_coefficient_effect(self):
        """Test different coefficients."""
        audio = np.random.randn(1000)
        
        # Higher coefficient = stronger filter
        filtered_weak = apply_preemphasis(audio, coef=0.9)
        filtered_strong = apply_preemphasis(audio, coef=0.99)
        
        # Both should have same length
        assert len(filtered_weak) == len(filtered_strong) == len(audio)


class TestChunkValidation:
    """Test energy-based chunk validation."""
    
    def test_is_valid_chunk_silence(self):
        """Test that silence is rejected."""
        silent_chunk = np.zeros(1000)
        assert not is_valid_chunk(silent_chunk, energy_threshold=0.01)
        
    def test_is_valid_chunk_noise(self):
        """Test that normal audio is accepted."""
        noisy_chunk = np.random.randn(1000) * 0.1
        assert is_valid_chunk(noisy_chunk, energy_threshold=0.01)
        
    def test_is_valid_chunk_very_quiet(self):
        """Test that very quiet audio is rejected."""
        quiet_chunk = np.random.randn(1000) * 0.001  # Very quiet
        assert not is_valid_chunk(quiet_chunk, energy_threshold=0.01)
        
    def test_is_valid_chunk_threshold(self):
        """Test different thresholds."""
        chunk = np.random.randn(1000) * 0.05  # Medium energy
        
        # Should pass with low threshold
        assert is_valid_chunk(chunk, energy_threshold=0.01)
        
        # Should fail with high threshold
        assert not is_valid_chunk(chunk, energy_threshold=0.1)


class TestParameterConfiguration:
    """Test that configuration is correctly applied."""
    
    def test_optimized_parameters_in_use(self):
        """Verify that optimized parameters are being used."""
        # Import after module initialization
        from Feature_Extraction import USE_OPTIMIZED_PARAMETERS
        
        if USE_OPTIMIZED_PARAMETERS:
            assert N_FFT == 1024, f"Expected N_FFT=1024, got {N_FFT}"
            assert HOP_LENGTH == 256, f"Expected HOP_LENGTH=256, got {HOP_LENGTH}"
            assert N_MELS == 80, f"Expected N_MELS=80, got {N_MELS}"
            print("  ✓ Using OPTIMIZED parameters")
        else:
            assert N_FFT == 2048, f"Expected N_FFT=2048, got {N_FFT}"
            assert HOP_LENGTH == 512, f"Expected HOP_LENGTH=512, got {HOP_LENGTH}"
            assert N_MELS == 128, f"Expected N_MELS=128, got {N_MELS}"
            print("  ✓ Using LEGACY parameters")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
