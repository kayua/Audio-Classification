"""
Mosquito Audio Organizer and Processor
=======================================
This script reads mosquito recording data from an Excel file, processes audio files,
and organizes them into a hierarchical folder structure.

Hierarchy: Species -> Sex -> Quantity
Processing: Format conversion -> Low-pass filter -> Resampling to 8kHz

Dependencies:
    - librosa (audio loading and processing)
    - soundfile (WAV file writing)
    - scipy (signal processing)
    - numpy (numerical operations)
    - pandas (Excel reading)
    - openpyxl (Excel file support)
    - tqdm (progress bars)
    - matplotlib (spectrogram visualization)
    - audioread or ffmpeg (for .3gp and other exotic formats)

Installation:
    pip install librosa soundfile scipy numpy pandas openpyxl tqdm matplotlib

    For .3gp support, also install:
    - On Ubuntu/Debian: sudo apt-get install ffmpeg
    - On macOS: brew install ffmpeg
    - On Windows: Download from https://ffmpeg.org/download.html
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
try:
    import librosa
    import librosa.display
    import soundfile as sf
    from scipy import signal
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install librosa soundfile scipy numpy pandas openpyxl tqdm matplotlib")
    exit(1)

from tqdm import tqdm


class MosquitoAudioOrganizer:
    """
    Main class for organizing and processing mosquito audio recordings.

    Attributes:
        excel_path (Path): Path to the Excel file containing recording metadata
        dataset_path (Path): Path to the folder containing raw audio files
        output_path (Path): Path where organized files will be saved
        target_sr (int): Target sampling rate (8000 Hz)
        cutoff_freq (int): Low-pass filter cutoff frequency (4000 Hz)
    """

    def __init__(self, excel_path: str, dataset_path: str, output_path: str):
        """
        Initialize the audio organizer.

        Args:
            excel_path: Path to Excel file with metadata
            dataset_path: Path to folder with audio files
            output_path: Path to output organized files
        """
        self.excel_path = Path(excel_path)
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_sr = 8000  # Target sampling rate: 8 kHz
        self.cutoff_freq = 4000  # Low-pass filter cutoff: 4 kHz

        # Setup logging
        self._setup_logging()

        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

    def _setup_logging(self):
        """Configure logging with detailed formatting."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('mosquito_audio_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("Mosquito Audio Organizer - Starting")
        self.logger.info("="*70)

    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate the Excel file containing recording metadata.

        Returns:
            DataFrame with recording metadata

        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If required columns are missing
        """
        self.logger.info(f"Loading metadata from: {self.excel_path}")

        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

        # Read Excel file
        try:
            df = pd.read_excel(self.excel_path)
            self.logger.info(f"Successfully loaded {len(df)} records")
        except Exception as e:
            self.logger.error(f"Failed to read Excel file: {e}")
            raise

        # Log all columns found in the Excel file
        self.logger.info(f"Columns found in Excel: {list(df.columns)}")

        # Map flexible column names to standardized names
        column_mapping = self._find_columns(df.columns)

        # Rename columns to standard names
        df = df.rename(columns=column_mapping)
        self.logger.info(f"Mapped columns: {column_mapping}")

        # Validate required columns
        required_columns = ['id_gravacao', 'especie_mosquito', 'sexo_mosquitos', 'quantidade_mosquitos']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.error(f"Available columns after mapping: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean data
        df = df.dropna(subset=required_columns)
        self.logger.info(f"After cleaning: {len(df)} valid records")

        # Log species distribution
        species_counts = df['especie_mosquito'].value_counts()
        self.logger.info(f"Species distribution:")
        for species, count in species_counts.items():
            self.logger.info(f"  - {species}: {count} recordings")

        return df

    def _find_columns(self, columns: List[str]) -> Dict[str, str]:
        """
        Find and map column names flexibly (handles suffixes and variations).

        Args:
            columns: List of column names from Excel

        Returns:
            Dictionary mapping original column names to standardized names
        """
        mapping = {}

        # Required column patterns (case-insensitive, handles suffixes and underscores)
        patterns = {
            'id_gravacao': ['id_gravacao', 'id gravacao', 'idgravacao'],
            'especie_mosquito': ['especie_mosquito', 'especie mosquito', 'especie'],
            'sexo_mosquitos': ['sexo_mosquitos', 'sexo mosquitos', 'sexo'],
            'quantidade_mosquitos': ['quantidade_mosquitos', 'quantidade mosquitos', 'quantidade']
        }

        for standard_name, possible_names in patterns.items():
            found = False
            for col in columns:
                col_normalized = col.lower().strip().replace('_', ' ')
                # Check if column matches any pattern
                for pattern in possible_names:
                    pattern_normalized = pattern.lower().replace('_', ' ')
                    if col_normalized.startswith(pattern_normalized) or col_normalized == pattern_normalized:
                        mapping[col] = standard_name
                        found = True
                        break
                if found:
                    break

        return mapping

    def find_audio_file(self, file_id: str) -> Path:
        """
        Search for audio file in dataset folder with various extensions.

        Args:
            file_id: Recording ID to search for (may include extension)

        Returns:
            Path to the audio file if found

        Raises:
            FileNotFoundError: If file is not found in any supported format
        """
        # Remove extension from file_id if present
        file_base = file_id.rsplit('.', 1)[0] if '.' in file_id else file_id

        # Common audio extensions
        extensions = ['.3gp', '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.mp4']

        # First try with the exact filename (in case it includes extension)
        if '.' in file_id:
            file_path = self.dataset_path / file_id
            if file_path.exists():
                return file_path

        # Try with different extensions
        for ext in extensions:
            file_path = self.dataset_path / f"{file_base}{ext}"
            if file_path.exists():
                return file_path

        # Try recursive search in subfolders
        for file_path in self.dataset_path.rglob(f"{file_base}*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                return file_path

        # Last attempt: search for any file containing the base name
        for file_path in self.dataset_path.rglob(f"*{file_base}*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                self.logger.warning(f"Found file with different name: {file_path.name}")
                return file_path

        raise FileNotFoundError(f"Audio file not found: {file_id}")

    def apply_lowpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply low-pass filter to attenuate frequencies above cutoff.

        Uses a Butterworth filter for smooth frequency response.

        Args:
            audio: Audio signal as numpy array
            sr: Current sampling rate

        Returns:
            Filtered audio signal
        """
        # Design Butterworth low-pass filter (4th order)
        nyquist = sr / 2
        normalized_cutoff = self.cutoff_freq / nyquist

        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            self.logger.warning(f"Cutoff frequency {self.cutoff_freq} Hz is too high for sampling rate {sr} Hz")
            return audio

        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)

        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)

        return filtered_audio

    def generate_spectrogram(self, audio: np.ndarray, sr: int, output_path: Path) -> bool:
        """
        Generate and save spectrogram visualization with linear frequency scale.

        Creates a spectrogram with linear frequency axis (0 to 4096 Hz) with orange/red color scheme.

        Args:
            audio: Audio signal as numpy array
            sr: Sampling rate
            output_path: Path where spectrogram image will be saved

        Returns:
            True if generation succeeded, False otherwise
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

            # Compute STFT (Short-Time Fourier Transform) for linear frequency scale
            n_fft = 2048
            hop_length = 512

            # Compute spectrogram
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Display spectrogram with linear frequency scale
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='linear',  # Linear frequency scale
                ax=ax,
                cmap='hot'  # Orange/red color scheme
            )

            # Set y-axis limits to 0-4096 Hz
            ax.set_ylim([0, 4096])

            # Set custom y-axis ticks for linear scale
            yticks = np.arange(0, 4097, 256)  # 0, 256, 512, 768, 1024, ..., 4096
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(y)) for y in yticks])

            # Labels
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('')

            # Add colorbar
            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.set_label('Intensity (dB)')

            # Tight layout
            plt.tight_layout()

            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            return True

        except Exception as e:
            self.logger.error(f"Error generating spectrogram: {str(e)}")
            plt.close('all')
            return False

    def generate_histogram(self, audio: np.ndarray, sr: int, output_path: Path) -> bool:
        """
        Generate and save frequency intensity histogram with triangular weighted binning.

        Creates a histogram showing the mean intensity of frequency bands over time.
        Uses triangular weighting to smooth adjacent frequencies (4 bins per bar).

        Args:
            audio: Audio signal as numpy array
            sr: Sampling rate
            output_path: Path where histogram image will be saved

        Returns:
            True if generation succeeded, False otherwise
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

            # Compute STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Calculate mean intensity for each frequency bin over time
            mean_intensity = np.mean(S_db, axis=1)

            # Create frequency bins (linear scale up to Nyquist frequency)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Filter to show only up to 4096 Hz
            max_freq_idx = np.where(frequencies <= 4096)[0][-1] if np.any(frequencies <= 4096) else len(frequencies)
            frequencies_filtered = frequencies[:max_freq_idx + 1]
            mean_intensity_filtered = mean_intensity[:max_freq_idx + 1]

            # Discretization with triangular weighting
            bin_size = 4  # Changed from 8 to 4
            num_bars = len(frequencies_filtered) // bin_size

            # Create triangular window weights (peak at center)
            # For 4 points: [1, 2, 2, 1]
            triangular_weights = np.array([1, 2, 2, 1], dtype=float)
            triangular_weights /= triangular_weights.sum()  # Normalize

            # Apply triangular weighting to create discretized bars
            discretized_intensities = []
            discretized_frequencies = []

            for i in range(num_bars):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size

                if end_idx <= len(mean_intensity_filtered):
                    # Apply triangular weighting
                    intensity_segment = mean_intensity_filtered[start_idx:end_idx]
                    weighted_intensity = np.sum(intensity_segment * triangular_weights)
                    discretized_intensities.append(weighted_intensity)

                    # Center frequency of this bar
                    center_freq = frequencies_filtered[start_idx + bin_size // 2]
                    discretized_frequencies.append(center_freq)

            discretized_intensities = np.array(discretized_intensities)
            discretized_frequencies = np.array(discretized_frequencies)

            # Invert Y axis: transform dB values so higher intensity = higher value
            # Convert from negative dB to positive scale (more intensity = higher bar)
            discretized_intensities_inverted = discretized_intensities - np.min(discretized_intensities)

            # Calculate bar width
            bar_width = frequencies_filtered[bin_size] - frequencies_filtered[0]

            # Plot as bar chart with orange color from bottom
            ax.bar(discretized_frequencies, discretized_intensities_inverted,
                   width=bar_width, color='#FF6B35', alpha=0.9, edgecolor='none',
                   bottom=None)  # Ensures bars start from the bottom

            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Relative Intensity')
            ax.set_title('Average Frequency Intensity Over Time (Triangular Weighted)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim([0, 4096])

            # Set custom x-axis ticks
            xticks = np.arange(0, 4097, 512)
            ax.set_xticks(xticks)

            # Tight layout
            plt.tight_layout()

            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            self.logger.debug(
                f"Generated histogram with {len(discretized_intensities)} bars (triangular weighted, bin_size=4)")

            return True

        except Exception as e:
            self.logger.error(f"Error generating histogram: {str(e)}")
            plt.close('all')
            return False

    @staticmethod
    def get_sex_label(sex_code: str) -> str:
        """
        Convert sex code to folder label.

        Args:
            sex_code: Sex code from Excel (M, F, or other)

        Returns:
            Folder label (Male, Female, or Both)
        """
        sex_code = str(sex_code).strip().upper()

        if sex_code == 'M':
            return 'Male'
        elif sex_code == 'F':
            return 'Female'
        else:
            return 'Both'

    def create_folder_structure(self, species: str, sex: str, quantity: int) -> Dict[str, Path]:
        """
        Create hierarchical folder structure: Species/Sex/Quantity/[sounds, spectrograms, histograms].

        Args:
            species: Mosquito species name
            sex: Sex code (M/F)
            quantity: Number of specimens

        Returns:
            Dictionary with paths for sounds, spectrograms, and histograms folders
        """
        # Clean species name for folder (replace spaces with underscores)
        species_folder = species.strip().replace(' ', '_').replace('/', '_')

        # Get sex label
        sex_folder = self.get_sex_label(sex)

        # Create quantity folder
        quantity_folder = f"Quantity_{int(quantity)}"

        # Build base path
        base_path = self.output_path / species_folder / sex_folder / quantity_folder

        # Create subdirectories
        paths = {
            'sounds': base_path / 'sounds',
            'spectrograms': base_path / 'spectrograms',
            'histograms': base_path / 'histograms'
        }

        # Create all directories
        for folder_path in paths.values():
            folder_path.mkdir(parents=True, exist_ok=True)

        return paths

    def process_all(self):
        """
        Main processing function: load metadata and process all audio files.

        This function orchestrates the entire workflow:
        1. Load metadata from Excel
        2. Process each recording
        3. Generate spectrograms and histograms
        4. Generate summary statistics
        """
        try:
            # Load metadata
            df = self.load_metadata()
            self.stats['total_files'] = len(df)

            # Verify dataset folder exists
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset folder not found: {self.dataset_path}")

            self.logger.info(f"Dataset folder: {self.dataset_path}")
            self.logger.info(f"Output folder: {self.output_path}")
            self.logger.info(f"Processing {len(df)} recordings...")

            # Process each recording
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
                file_id = row['id_gravacao']
                species = row['especie_mosquito']
                sex = row['sexo_mosquitos']
                quantity = row['quantidade_mosquitos']

                try:
                    # Find audio file
                    input_path = self.find_audio_file(file_id)

                    # Create target folder structure
                    target_folders = self.create_folder_structure(species, sex, quantity)

                    # Define output paths
                    base_filename = file_id.rsplit('.', 1)[0] if '.' in file_id else file_id
                    audio_output = target_folders['sounds'] / f"{base_filename}.wav"
                    spectrogram_output = target_folders['spectrograms'] / f"{base_filename}_spectrogram.png"
                    histogram_output = target_folders['histograms'] / f"{base_filename}_histogram.png"

                    # Skip if all files already exist
                    if audio_output.exists() and spectrogram_output.exists() and histogram_output.exists():
                        self.logger.info(f"  → All files exist, skipping")
                        self.stats['skipped'] += 1
                        continue

                    # Load and process audio
                    self.logger.debug(f"Loading audio: {input_path.name}")
                    try:
                        original_audio, original_sr = librosa.load(input_path, sr=None, mono=True)
                    except Exception as e:
                        self.logger.debug(f"Standard loading failed, trying audioread backend: {e}")
                        original_audio, original_sr = librosa.load(str(input_path), sr=None, mono=True, res_type='kaiser_fast')

                    if len(original_audio) == 0:
                        raise ValueError("Empty audio file")

                    # Apply low-pass filter
                    self.logger.debug(f"Applying low-pass filter (cutoff: {self.cutoff_freq} Hz)")
                    filtered_audio = self.apply_lowpass_filter(original_audio, original_sr)

                    # Resample to target sampling rate
                    if original_sr != self.target_sr:
                        self.logger.debug(f"Resampling from {original_sr} Hz to {self.target_sr} Hz")
                        processed_audio = librosa.resample(filtered_audio, orig_sr=original_sr, target_sr=self.target_sr)
                    else:
                        processed_audio = filtered_audio

                    # Normalize audio
                    max_val = np.abs(processed_audio).max()
                    if max_val > 0:
                        processed_audio = processed_audio / max_val * 0.95

                    # Save processed audio
                    if not audio_output.exists():
                        sf.write(audio_output, processed_audio, self.target_sr, subtype='PCM_16')
                        self.logger.debug(f"Saved audio: {audio_output.name}")

                    # Generate spectrogram
                    if not spectrogram_output.exists():
                        self.logger.debug(f"Generating spectrogram")
                        self.generate_spectrogram(processed_audio, self.target_sr, spectrogram_output)
                        self.logger.debug(f"Saved spectrogram: {spectrogram_output.name}")

                    # Generate histogram
                    if not histogram_output.exists():
                        self.logger.debug(f"Generating histogram")
                        self.generate_histogram(processed_audio, self.target_sr, histogram_output)
                        self.logger.debug(f"Saved histogram: {histogram_output.name}")

                    self.stats['processed'] += 1

                except FileNotFoundError as e:
                    self.logger.warning(f"  → File not found: {file_id}")
                    self.stats['failed'] += 1

                except Exception as e:
                    self.logger.error(f"  → Error processing {file_id}: {str(e)}")
                    self.stats['failed'] += 1

            # Print summary
            self._print_summary()

        except Exception as e:
            self.logger.error(f"Fatal error during processing: {str(e)}")
            raise

    def _print_summary(self):
        """Print processing summary statistics."""
        self.logger.info("="*70)
        self.logger.info("Processing Summary")
        self.logger.info("="*70)
        self.logger.info(f"Total files:        {self.stats['total_files']}")
        self.logger.info(f"Successfully processed: {self.stats['processed']}")
        self.logger.info(f"Skipped (existing): {self.stats['skipped']}")
        self.logger.info(f"Failed:             {self.stats['failed']}")

        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed'] / self.stats['total_files']) * 100
            self.logger.info(f"Success rate:       {success_rate:.1f}%")

        self.logger.info("="*70)
        self.logger.info(f"Organized files saved in: {self.output_path}")
        self.logger.info("Processing complete!")


def main():
    """
    Main entry point for the script.

    Configure paths and run the audio organizer.

    The output structure will be:
    Organized_Mosquito_Audio/
    ├── species_name/
    │   ├── Male/
    │   │   └── Quantity_N/
    │   │       ├── sounds/           # Processed WAV files
    │   │       ├── spectrograms/     # Spectrogram visualizations
    │   │       └── histograms/       # Amplitude distribution histograms
    │   └── Female/
    │       └── Quantity_N/
    │           ├── sounds/
    │           ├── spectrograms/
    │           └── histograms/
    """
    # Configuration
    EXCEL_FILE = "gravacoes_2021_2022.xlsx"  # Your Excel file
    DATASET_FOLDER = "Dataset"               # Folder with audio files
    OUTPUT_FOLDER = "Organized_Mosquito_Audio"  # Output folder

    # Create organizer instance
    organizer = MosquitoAudioOrganizer(
        excel_path=EXCEL_FILE,
        dataset_path=DATASET_FOLDER,
        output_path=OUTPUT_FOLDER
    )

    # Process all files
    organizer.process_all()


if __name__ == "__main__":
    main()