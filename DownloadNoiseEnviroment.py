#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/10/16'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 unknown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
ESC-50 Environmental Sound Dataset Downloader and Processor
============================================================
This script downloads and processes the ESC-50 environmental sound dataset:
- ESC-50 (50 classes, 2000 clips)

Hierarchy: Dataset -> Class -> [sounds, spectrograms, histograms]
Processing: Format conversion -> Low-pass filter -> Resampling to 8kHz

Features:
    - Automatic dataset downloading and extraction
    - Audio processing with low-pass filter (4kHz cutoff)
    - Resampling to 8kHz
    - Spectrogram generation (linear scale, 0-4096 Hz)
    - Histogram generation (triangular weighted)
    - Comprehensive statistics (20+ metrics)
    - Progress tracking with tqdm

Dependencies:
    pip install librosa soundfile scipy numpy pandas openpyxl tqdm matplotlib requests

    For audio format support:
    - Ubuntu/Debian: sudo apt-get install ffmpeg
    - macOS: brew install ffmpeg
    - Windows: Download from https://ffmpeg.org/download.html
"""

import os
import sys
import json
import shutil
import logging
import warnings
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_FOLDER = "ESC50_Processed"  # Main output folder
DOWNLOAD_FOLDER = "downloads"  # Temporary download folder
TARGET_SR = 8000  # Target sampling rate
CUTOFF_FREQ = 4000  # Low-pass filter cutoff


class DatasetDownloader:
    """Handles downloading and extracting ESC-50 dataset."""

    def __init__(self, download_path: Path):
        self.download_path = download_path
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def download_file(self, url: str, destination: Path, description: str = "Downloading"):
        """Download file with progress bar."""
        if destination.exists():
            self.logger.info(f"File already exists: {destination.name}")
            return True

        try:
            self.logger.info(f"Downloading {description} from {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(destination, 'wb') as f, tqdm(
                    desc=description,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            self.logger.info(f"Downloaded: {destination.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {description}: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file."""
        self.logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        self.logger.info(f"Extracted to {extract_to}")

    def download_esc50(self) -> Path:
        """Download ESC-50 dataset."""
        self.logger.info("=" * 70)
        self.logger.info("Downloading ESC-50 Dataset")
        self.logger.info("=" * 70)

        # ESC-50 GitHub releases
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        zip_path = self.download_path / "esc50.zip"
        extract_path = self.download_path / "ESC-50"

        if not self.download_file(url, zip_path, "ESC-50"):
            raise Exception("Failed to download ESC-50")

        if not extract_path.exists():
            self.extract_zip(zip_path, self.download_path)
            # Rename extracted folder
            extracted = self.download_path / "ESC-50-master"
            if extracted.exists():
                extracted.rename(extract_path)

        return extract_path


class ESC50Processor:
    """Process ESC-50 environmental sound dataset."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.target_sr = TARGET_SR
        self.cutoff_freq = CUTOFF_FREQ

        # Setup logging
        self._setup_logging()

        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

        # Detailed statistics
        self.detailed_stats = {
            'audio_metrics': [],
            'by_class': defaultdict(lambda: {
                'count': 0,
                'total_duration': 0,
                'durations': []
            }),
            'original_sample_rates': [],
            'processing_start_time': datetime.now().isoformat(),
            'processing_end_time': None
        }

    def _setup_logging(self):
        """Configure logging."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('esc50_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 70)
        self.logger.info("ESC-50 Processor - Starting")
        self.logger.info("=" * 70)

    def apply_lowpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply low-pass Butterworth filter."""
        nyquist = sr / 2
        normalized_cutoff = self.cutoff_freq / nyquist

        if normalized_cutoff >= 1.0:
            self.logger.warning(f"Cutoff frequency too high for SR {sr}")
            return audio

        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio

    def calculate_audio_metrics(self, audio: np.ndarray, sr: int) -> Dict:
        """Calculate comprehensive audio metrics."""
        metrics = {}

        metrics['duration_seconds'] = len(audio) / sr
        metrics['rms_amplitude'] = np.sqrt(np.mean(audio ** 2))
        metrics['peak_amplitude'] = np.max(np.abs(audio))
        metrics['zero_crossing_rate'] = np.mean(librosa.zero_crossings(audio))

        # Spectral features
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)

        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        metrics['spectral_centroid_mean'] = np.mean(spectral_centroids)
        metrics['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        metrics['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        metrics['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

        metrics['energy'] = np.sum(audio ** 2)
        metrics['dynamic_range_db'] = 20 * np.log10(
            metrics['peak_amplitude'] / (metrics['rms_amplitude'] + 1e-10))

        return metrics

    def generate_spectrogram(self, audio: np.ndarray, sr: int, output_path: Path) -> bool:
        """Generate spectrogram with linear frequency scale (0-4096 Hz)."""
        try:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

            n_fft = 2048
            hop_length = 512

            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(
                S_db, sr=sr, hop_length=hop_length,
                x_axis='time', y_axis='linear',
                ax=ax, cmap='hot'
            )

            ax.set_ylim([0, 4096])
            yticks = np.arange(0, 4097, 512)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(y)) for y in yticks])

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.set_label('Intensity (dB)')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            return True

        except Exception as e:
            self.logger.error(f"Error generating spectrogram: {e}")
            plt.close('all')
            return False

    def generate_histogram(self, audio: np.ndarray, sr: int, output_path: Path) -> bool:
        """Generate frequency intensity histogram with triangular weighting."""
        try:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

            n_fft = 2048
            hop_length = 512
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            mean_intensity = np.mean(S_db, axis=1)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            max_freq_idx = np.where(frequencies <= 4096)[0][-1] if np.any(frequencies <= 4096) else len(frequencies)
            frequencies_filtered = frequencies[:max_freq_idx + 1]
            mean_intensity_filtered = mean_intensity[:max_freq_idx + 1]

            # Triangular weighting
            bin_size = 4
            num_bars = len(frequencies_filtered) // bin_size
            triangular_weights = np.array([1, 2, 2, 1], dtype=float)
            triangular_weights /= triangular_weights.sum()

            discretized_intensities = []
            discretized_frequencies = []

            for i in range(num_bars):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size

                if end_idx <= len(mean_intensity_filtered):
                    intensity_segment = mean_intensity_filtered[start_idx:end_idx]
                    weighted_intensity = np.sum(intensity_segment * triangular_weights)
                    discretized_intensities.append(weighted_intensity)

                    center_freq = frequencies_filtered[start_idx + bin_size // 2]
                    discretized_frequencies.append(center_freq)

            discretized_intensities = np.array(discretized_intensities)
            discretized_frequencies = np.array(discretized_frequencies)

            # Invert Y axis
            discretized_intensities_inverted = discretized_intensities - np.min(discretized_intensities)

            bar_width = frequencies_filtered[bin_size] - frequencies_filtered[0]

            ax.bar(discretized_frequencies, discretized_intensities_inverted,
                   width=bar_width, color='#FF6B35', alpha=0.5, edgecolor='#FFFFFF')

            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Relative Intensity')
            ax.set_title('Average Frequency Intensity Over Time')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim([0, 4096])

            xticks = np.arange(0, 4097, 512)
            ax.set_xticks(xticks)

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            return True

        except Exception as e:
            self.logger.error(f"Error generating histogram: {e}")
            plt.close('all')
            return False

    def process_audio_file(self, input_path: Path, class_name: str, file_id: str):
        """Process a single audio file."""
        try:
            # Create folder structure
            class_folder = self.output_path / class_name
            sounds_folder = class_folder / 'sounds'
            spectrograms_folder = class_folder / 'spectrograms'
            histograms_folder = class_folder / 'histograms'

            for folder in [sounds_folder, spectrograms_folder, histograms_folder]:
                folder.mkdir(parents=True, exist_ok=True)

            # Define output paths
            base_filename = input_path.stem
            audio_output = sounds_folder / f"{base_filename}.wav"
            spectrogram_output = spectrograms_folder / f"{base_filename}_spectrogram.png"
            histogram_output = histograms_folder / f"{base_filename}_histogram.png"

            # Skip if all files exist
            if audio_output.exists() and spectrogram_output.exists() and histogram_output.exists():
                self.stats['skipped'] += 1
                return

            # Load audio
            original_audio, original_sr = librosa.load(input_path, sr=None, mono=True)

            if len(original_audio) == 0:
                raise ValueError("Empty audio file")

            # Apply low-pass filter
            filtered_audio = self.apply_lowpass_filter(original_audio, original_sr)

            # Resample
            if original_sr != self.target_sr:
                processed_audio = librosa.resample(filtered_audio, orig_sr=original_sr, target_sr=self.target_sr)
            else:
                processed_audio = filtered_audio

            # Normalize
            max_val = np.abs(processed_audio).max()
            if max_val > 0:
                processed_audio = processed_audio / max_val * 0.95

            # Save processed audio
            if not audio_output.exists():
                sf.write(audio_output, processed_audio, self.target_sr, subtype='PCM_16')

            # Generate visualizations
            if not spectrogram_output.exists():
                self.generate_spectrogram(processed_audio, self.target_sr, spectrogram_output)

            if not histogram_output.exists():
                self.generate_histogram(processed_audio, self.target_sr, histogram_output)

            # Collect statistics
            metrics = self.calculate_audio_metrics(processed_audio, self.target_sr)
            metrics['file_id'] = file_id
            metrics['class'] = class_name
            metrics['original_sample_rate'] = original_sr
            metrics['original_file_size_bytes'] = input_path.stat().st_size
            metrics['processed_file_size_bytes'] = audio_output.stat().st_size if audio_output.exists() else 0

            self.detailed_stats['audio_metrics'].append(metrics)

            self.detailed_stats['by_class'][class_name]['count'] += 1
            self.detailed_stats['by_class'][class_name]['total_duration'] += metrics['duration_seconds']
            self.detailed_stats['by_class'][class_name]['durations'].append(metrics['duration_seconds'])

            self.detailed_stats['original_sample_rates'].append(original_sr)

            self.stats['processed'] += 1

        except Exception as e:
            self.logger.error(f"Error processing {input_path.name}: {e}")
            self.stats['failed'] += 1

    def process_esc50(self, esc50_path: Path):
        """Process ESC-50 dataset."""
        self.logger.info("=" * 70)
        self.logger.info("Processing ESC-50 Dataset")
        self.logger.info("=" * 70)

        audio_path = esc50_path / 'audio'
        meta_path = esc50_path / 'meta' / 'esc50.csv'

        if not audio_path.exists() or not meta_path.exists():
            self.logger.error("ESC-50 structure not found")
            return

        # Load metadata
        meta_df = pd.read_csv(meta_path)

        self.logger.info(f"Found {len(meta_df)} files in ESC-50")

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing ESC-50"):
            filename = row['filename']
            category = row['category']

            audio_file = audio_path / filename

            if audio_file.exists():
                self.stats['total_files'] += 1
                self.process_audio_file(audio_file, category, filename)

    def generate_statistics_report(self):
        """Generate comprehensive statistics report."""
        self.logger.info("=" * 70)
        self.logger.info("Generating Statistics Report")
        self.logger.info("=" * 70)

        self.detailed_stats['processing_end_time'] = datetime.now().isoformat()

        all_metrics = self.detailed_stats['audio_metrics']

        if not all_metrics:
            self.logger.warning("No metrics collected")
            return

        df = pd.DataFrame(all_metrics)

        # Prepare summary
        summary = {
            'processing_info': {
                'start_time': self.detailed_stats['processing_start_time'],
                'end_time': self.detailed_stats['processing_end_time'],
                'total_files': self.stats['total_files'],
                'processed': self.stats['processed'],
                'failed': self.stats['failed'],
                'skipped': self.stats['skipped']
            },
            'overall_statistics': {
                'total_audio_count': len(all_metrics),
                'total_duration_seconds': float(df['duration_seconds'].sum()),
                'total_duration_hours': float(df['duration_seconds'].sum() / 3600),
                'average_duration_seconds': float(df['duration_seconds'].mean()),
                'median_duration_seconds': float(df['duration_seconds'].median()),
                'min_duration_seconds': float(df['duration_seconds'].min()),
                'max_duration_seconds': float(df['duration_seconds'].max())
            },
            'statistics_by_class': {}
        }

        # By class
        for class_name, data in self.detailed_stats['by_class'].items():
            summary['statistics_by_class'][class_name] = {
                'count': data['count'],
                'total_duration_seconds': data['total_duration'],
                'average_duration_seconds': data['total_duration'] / (data['count'] + 1e-10),
                'min_duration': float(min(data['durations'])) if data['durations'] else 0,
                'max_duration': float(max(data['durations'])) if data['durations'] else 0
            }

        # Save JSON
        json_output = self.output_path / 'statistics_report.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON statistics saved to: {json_output}")

        # Save CSV
        csv_output = self.output_path / 'detailed_audio_metrics.csv'
        df.to_csv(csv_output, index=False, encoding='utf-8')
        self.logger.info(f"CSV metrics saved to: {csv_output}")

        # Print summary
        self.logger.info("=" * 70)
        self.logger.info("Processing Summary")
        self.logger.info("=" * 70)
        self.logger.info(f"Total files: {self.stats['total_files']}")
        self.logger.info(f"Processed: {self.stats['processed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Total duration: {summary['overall_statistics']['total_duration_hours']:.2f} hours")
        self.logger.info("=" * 70)


def main():
    """Main entry point."""
    print("=" * 70)
    print("ESC-50 Dataset Processor")
    print("=" * 70)

    # Step 1: Download ESC-50
    downloader = DatasetDownloader(Path(DOWNLOAD_FOLDER))

    print("\n[1/3] Downloading ESC-50 dataset...")
    esc50_path = downloader.download_esc50()

    # Step 2: Process ESC-50
    print("\n[2/3] Processing audio files...")
    processor = ESC50Processor(OUTPUT_FOLDER)

    if esc50_path and esc50_path.exists():
        processor.process_esc50(esc50_path)
    else:
        print("ERROR: ESC-50 dataset not found!")
        return

    # Step 3: Generate statistics
    print("\n[3/3] Generating statistics...")
    processor.generate_statistics_report()

    # Done
    print("\nProcessing complete!")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FOLDER}/statistics_report.json")
    print(f"  - {OUTPUT_FOLDER}/detailed_audio_metrics.csv")
    print("\nFolder structure:")
    print(f"  {OUTPUT_FOLDER}/")
    print(f"    └── [class_name]/")
    print(f"        ├── sounds/")
    print(f"        ├── spectrograms/")
    print(f"        └── histograms/")


if __name__ == "__main__":
    main()