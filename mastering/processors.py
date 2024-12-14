import os
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
from django.conf import settings
import noisereduce as nr
from scipy.fftpack import fft, ifft

class SmarterVocalMasteringProcessor:
    def __init__(
        self, 
        target_loudness=-14, 
        sample_rate=44100, 
        dynamic_range_threshold=6
    ):
        self.target_loudness = target_loudness
        self.sample_rate = sample_rate
        self.dynamic_range_threshold = dynamic_range_threshold

    # 1. Noise Profiling and Adaptive Noise Reduction
    def noise_reduction(self, audio):
        try:
            # Ensure non-zero audio values to avoid computational issues
            audio = np.clip(audio, a_min=1e-10, a_max=None)

            # Extract a noise sample from the first 1 second for a more representative profile
            noise_duration = int(1.0 * self.sample_rate)  # 1 second
            noise_sample = audio[:noise_duration]

            # Check if the noise sample has sufficient variability
            if np.std(noise_sample) < 1e-3:
                print("Warning: Noise sample may not represent actual noise. Proceeding with caution.")

            # Perform noise reduction with dynamic parameters
            reduced_audio = nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=self.sample_rate,
                prop_decrease=0.5,  # Slightly higher suppression for better noise removal
                stationary=False    # Non-stationary mode for fluctuating noise
            )

            # Smooth transitions to avoid artifacts
            reduced_audio = np.clip(reduced_audio, -1.0, 1.0)  # Normalize after processing

            return reduced_audio
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio

    # 2. Dynamic Lookahead Compression
    def dynamic_compression(self, audio):
        try:
            threshold = -18  # dBFS
            ratio = 4  # Compression ratio
            attack = 0.01  # Seconds
            release = 0.1  # Seconds

            audio_db = 20 * np.log10(np.maximum(np.abs(audio), 1e-5))
            gain = np.ones_like(audio)

            above_threshold = audio_db > threshold
            gain[above_threshold] = 10 ** ((threshold - audio_db[above_threshold]) * (1 - 1 / ratio) / 20)

            # Smooth the gain using a moving average
            window_size = int(self.sample_rate * (attack + release))
            if window_size > 1:
                smoothed_gain = np.convolve(gain, np.ones(window_size) / window_size, mode='same')
            else:
                smoothed_gain = gain

            return audio * smoothed_gain
        except Exception as e:
            print(f"Compression error: {e}")
            return audio
      
    def equalize(self, audio):
        try:
            fft_data = fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1 / self.sample_rate)
            perceptual_curve = np.interp(np.abs(freqs), [100, 1000, 5000, 20000], [1.0, 1.2, 1.0, 0.8])
            return ifft(fft_data * perceptual_curve).real
        except Exception as e:
            print(f"Equalization error: {e}")
            return audio

    # 3. Phase Alignment
    def phase_alignment(self, audio):
        try:
            if audio.ndim > 1:
                left, right = audio[0], audio[1]
                delay = np.correlate(left, right, mode='full')
                alignment_offset = np.argmax(delay) - len(right) + 1
                if alignment_offset > 0:
                    right = np.pad(right, (alignment_offset, 0), mode='constant')[:len(left)]
                else:
                    left = np.pad(left, (-alignment_offset, 0), mode='constant')[:len(right)]
                return np.stack([left, right])
            return audio
        except Exception as e:
            print(f"Phase alignment error: {e}")
            return audio

    # 4. Psychoacoustic Modeling
    def psychoacoustic_enhancement(self, audio):
        try:
            spectrum = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
            masking_curve = np.ones_like(freqs)
            masking_curve[(freqs >= 2000) & (freqs <= 5000)] *= 1.2
            return np.fft.irfft(spectrum * masking_curve)
        except Exception as e:
            print(f"Psychoacoustic enhancement error: {e}")
            return audio

    # 5. Transient Shaping
    def transient_shaping(self, audio):
        try:
            # Calculate RMS envelope with a frame length of 2048 samples
            frame_length = 2048
            hop_length = frame_length // 2  # Overlap for smoother envelope
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

            # Normalize the RMS envelope to the range [0.8, 1.2] for gain control
            gain = np.interp(rms, (rms.min(), rms.max()), (0.8, 1.2))

            # Upsample gain to match the length of the audio
            gain_upsampled = np.interp(
                np.arange(len(audio)),
                np.linspace(0, len(audio), len(gain)),
                gain
            )

            # Apply gain to audio for transient shaping
            shaped_audio = audio * gain_upsampled
            return shaped_audio
        except Exception as e:
            print(f"Transient shaping error: {e}")
            return audio

    # 6. Mid/Side (M/S) Processing Refinement
    def mid_side_processing(self, audio):
        try:
            if audio.ndim > 1:
                left, right = audio[0], audio[1]
                mid = (left + right) / 2
                side = (left - right) / 2
                side *= 1.2  # Widen high frequencies
                return np.stack([mid + side, mid - side])
            return audio
        except Exception as e:
            print(f"M/S processing error: {e}")
            return audio

    # 7. Advanced De-Essing
    def de_essing(self, audio):
        try:
            D = librosa.stft(audio)
            mag, phase = np.abs(D), np.angle(D)
            high_freq_mask = mag > np.percentile(mag, 95)
            attenuation = np.clip(1 - (mag / mag.max()), 0.5, 1)
            mag[high_freq_mask] *= attenuation[high_freq_mask]
            return librosa.istft(mag * np.exp(1j * phase))
        except Exception as e:
            print(f"De-essing error: {e}")
            return audio

    # 8. Intelligent Harmonic Exciter
    def harmonic_exciter(self, audio):
        try:
            harmonic_audio = audio + 0.01 * np.tanh(audio * 2)
            return harmonic_audio
        except Exception as e:
            print(f"Harmonic exciter error: {e}")
            return audio

    # 9. Multiband Compression
    def multiband_compression(self, audio):
        try:
            bands = [(20, 200), (200, 1000), (1000, 5000), (5000, 20000)]
            compressed_audio = np.zeros_like(audio)
            for low, high in bands:
                sos = signal.butter(4, [low, high], btype='band', fs=self.sample_rate, output='sos')
                band = signal.sosfilt(sos, audio)
                gain = 0.9  # Example fixed gain, could be adaptive
                compressed_audio += band * gain
            return compressed_audio
        except Exception as e:
            print(f"Multiband compression error: {e}")
            return audio

    # 10. Loudness Range Optimization
    def loudness_normalization(self, audio):
        try:
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio)
            return pyln.normalize.loudness(audio, loudness, self.target_loudness)
        except Exception as e:
            print(f"Loudness normalization error: {e}")
            return audio
        
    # 11. Stereo Enhancement
    def stereo_enhancement(self, audio):
        try:
            # Ensure audio is stereo
            if audio.ndim == 1:
                raise ValueError("Input audio must be stereo for stereo enhancement.")

            # Separate mid (center) and side (stereo) channels
            mid = (audio[0, :] + audio[1, :]) / 2  # Mono channel (mid)
            side = (audio[0, :] - audio[1, :]) / 2  # Stereo difference (side)

            # Amplify the side channel for stereo widening
            side_widened = side * 1.2  # Adjust factor for widening effect

            # Recombine mid and side channels
            left = mid + side_widened
            right = mid - side_widened

            # Normalize to avoid clipping
            left = np.clip(left, -1.0, 1.0)
            right = np.clip(right, -1.0, 1.0)

            # Combine left and right channels into stereo
            enhanced_audio = np.vstack([left, right])

            return enhanced_audio
        except Exception as e:
            print(f"Stereo enhancement error: {e}")
            return audio

    def normalize_vocal_loudness(self, audio):
        try:
           # Calculate the loudness of the audio
            audio_db = librosa.amplitude_to_db(np.abs(audio))
            target_db=-20
            # Identify vocal segments (this is a simplified approach)
            # You might want to use a more sophisticated method to isolate vocals
            vocal_mask = audio_db > (np.mean(audio_db) + 10)  # Adjust threshold as needed
            vocal_segments = audio[vocal_mask]

            # Calculate the current loudness of the vocal segments
            current_loudness = librosa.amplitude_to_db(np.mean(np.abs(vocal_segments)))

            # Calculate the gain needed to reach the target loudness
            gain = target_db - current_loudness

            # Apply gain to the vocal segments
            normalized_audio = audio * (10 ** (gain / 20))

            return normalized_audio

        except Exception as e:
            print(f"Error in loudness normalization: {e}")
            return audio

    # 12. Export and Delivery Optimization
    def export_audio(self, audio, output_path):
        try:
            # sf.write(output_path, audio, self.sample_rate, subtype="PCM_16")

            # Convert the audio to 24-bit integer format
            audio_24bit = (audio * (2**23 - 1)).astype(np.int32)
            sf.write(output_path, audio, self.sample_rate, subtype="PCM_24")
            print(f"Audio exported successfully")
        except Exception as e:
            print(f"Export error: {e}")

    def process_vocal(self, input_path, output_path=None):
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            audio, sr = librosa.load(input_path, sr=self.sample_rate)
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            steps = [
                self.normalize_vocal_loudness,
                self.phase_alignment,
                self.de_essing,
                self.dynamic_compression,
                self.equalize,
                self.psychoacoustic_enhancement,
                self.multiband_compression,
                self.transient_shaping,
                self.harmonic_exciter,
                self.mid_side_processing,
                # self.stereo_enhancement, 
                self.loudness_normalization,
            ]

            for step in steps:
                audio = step(audio)

            if output_path is None:
                output_path = os.path.join(
                    settings.MEDIA_ROOT, 
                    'processed_vocals', 
                    f'processed_{os.path.basename(input_path)}'
                )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.export_audio(audio, output_path)
            return output_path
        except Exception as e:
            print(f"Processing error: {e}")
            return None