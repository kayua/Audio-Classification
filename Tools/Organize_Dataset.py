#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/10/16'
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
Dataset Organizer for Mosquito and Environmental Sound Classification
======================================================================
This script organizes processed audio files from mosquito and ESC-50 datasets
into a unified classification structure:

Directory Structure:
    dataset/
        ├── 0/  - Aedes aegypti Male
        ├── 1/  - Aedes aegypti Female
        ├── 2/  - Aedes albopictus (both sexes) + non-Aedes species
        └── 3/  - Environmental sounds (ALL ESC-50 categories)

Input Sources:
    - Organized_Mosquito_Audio/  (from mosquito processing script)
    - ESC50_Processed/           (from ESC-50 processing script)

Features:
    - Automatic file copying (preserves originals)
    - Progress tracking with tqdm
    - Detailed statistics reporting
    - Duplicate detection and handling
    - Comprehensive logging
    - Includes ALL ESC-50 categories in class 3

Usage:
    python dataset_organizer.py

Dependencies:
    pip install tqdm pandas
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Configuration
MOSQUITO_INPUT = "Organized_Mosquito_Audio"  # Output from mosquito script
ESC50_INPUT = "ESC50_Processed"  # Output from ESC-50 script
DATASET_OUTPUT = "dataset"  # Final dataset folder


class DatasetOrganizer:
    """
    Organizes processed audio files into classification dataset structure.

    Class mapping:
        0: Aedes aegypti Male
        1: Aedes aegypti Female
        2: Aedes albopictus (both sexes) + other mosquito species
        3: Environmental sounds (ALL ESC-50 categories)
    """

    def __init__(self, mosquito_path: str, esc50_path: str, output_path: str):
        """
        Initialize dataset organizer.

        Args:
            mosquito_path: Path to processed mosquito audio folder
            esc50_path: Path to processed ESC-50 folder
            output_path: Path to output dataset folder
        """
        self.mosquito_path = Path(mosquito_path)
        self.esc50_path = Path(esc50_path)
        self.output_path = Path(output_path)

        # Setup logging
        self._setup_logging()

        # Statistics
        self.stats = {
            'class_0': {'count': 0, 'sources': []},
            'class_1': {'count': 0, 'sources': []},
            'class_2': {'count': 0, 'sources': []},
            'class_3': {'count': 0, 'sources': [], 'categories': {}},
            'duplicates': 0,
            'errors': 0
        }

        # Track copied files to avoid duplicates
        self.copied_files = set()

    def _setup_logging(self):
        """Configure logging with detailed formatting."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('dataset_organizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 70)
        self.logger.info("Dataset Organizer - Starting")
        self.logger.info("=" * 70)

    def create_dataset_structure(self):
        """Create dataset folder structure with class subdirectories."""
        self.logger.info("Creating dataset structure...")

        for class_id in range(4):
            class_folder = self.output_path / str(class_id)
            class_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created: {class_folder}")

    def copy_audio_file(self, source: Path, destination: Path, class_id: int, source_info: str):
        """
        Copy audio file to destination with duplicate checking.

        Args:
            source: Source audio file path
            destination: Destination path
            class_id: Target class ID (0-3)
            source_info: Description of source for logging
        """
        try:
            # Check if file already exists with same name
            if destination.exists():
                self.logger.debug(f"File already exists: {destination.name}")
                self.stats['duplicates'] += 1
                return

            # Check if we already copied this exact file
            file_signature = f"{source.stem}_{source.stat().st_size}"
            if file_signature in self.copied_files:
                self.logger.debug(f"Duplicate detected: {source.name}")
                self.stats['duplicates'] += 1
                return

            # Copy file
            shutil.copy2(source, destination)
            self.copied_files.add(file_signature)

            # Update statistics
            self.stats[f'class_{class_id}']['count'] += 1
            self.stats[f'class_{class_id}']['sources'].append(source_info)

            # Track ESC-50 categories for class 3
            if class_id == 3 and source_info.startswith('ESC-50/'):
                category = source_info.split('/')[-1]
                if 'categories' in self.stats['class_3']:
                    if category not in self.stats['class_3']['categories']:
                        self.stats['class_3']['categories'][category] = 0
                    self.stats['class_3']['categories'][category] += 1

            self.logger.debug(f"Copied: {source.name} -> class_{class_id}/")

        except Exception as e:
            self.logger.error(f"Error copying {source.name}: {e}")
            self.stats['errors'] += 1

    def process_mosquito_audios(self):
        """Process mosquito audio files from Organized_Mosquito_Audio."""
        self.logger.info("=" * 70)
        self.logger.info("Processing Mosquito Audio Files")
        self.logger.info("=" * 70)

        if not self.mosquito_path.exists():
            self.logger.error(f"Mosquito audio folder not found: {self.mosquito_path}")
            return

        # Navigate through species folders
        species_folders = [f for f in self.mosquito_path.iterdir() if f.is_dir()]

        for species_folder in tqdm(species_folders, desc="Processing species"):
            species_name = species_folder.name.lower()

            # Navigate through sex folders
            sex_folders = [f for f in species_folder.iterdir() if f.is_dir()]

            for sex_folder in sex_folders:
                sex_name = sex_folder.name.lower()

                # Navigate through quantity folders
                quantity_folders = [f for f in sex_folder.iterdir() if f.is_dir()]

                for quantity_folder in quantity_folders:
                    # Get sounds folder
                    sounds_folder = quantity_folder / 'sounds'

                    if not sounds_folder.exists():
                        continue

                    # Get all WAV files
                    audio_files = list(sounds_folder.glob('*.wav'))

                    # Determine target class based on species and sex
                    class_id = self._determine_class(species_name, sex_name)

                    if class_id is None:
                        self.logger.warning(f"Unknown category: {species_name}/{sex_name}")
                        continue

                    # Copy files
                    class_folder = self.output_path / str(class_id)

                    for audio_file in audio_files:
                        destination = class_folder / audio_file.name
                        source_info = f"{species_name}/{sex_name}"
                        self.copy_audio_file(audio_file, destination, class_id, source_info)

    def _determine_class(self, species: str, sex: str) -> int:
        """
        Determine target class based on species and sex.

        Args:
            species: Species name (lowercase)
            sex: Sex name (lowercase)

        Returns:
            Class ID (0-2) or None if unknown
        """
        # Class 0: Aedes aegypti Male
        if 'aedes_aegypti' in species and 'male' in sex and 'female' not in sex:
            return 0

        # Class 1: Aedes aegypti Female
        if 'aedes_aegypti' in species and 'female' in sex and 'male' not in sex:
            return 1

        # Class 2: Aedes albopictus (both sexes) or other species
        # This includes:
        # - aedes_albopictus (male, female, or both)
        # - Any mixed aedes_aegypti recordings (male-female)
        # - Any other mosquito species
        if 'albopictus' in species:
            return 2

        if 'aedes_aegypti' in species and 'male' in sex and 'female' in sex:
            # Mixed sex recordings go to class 2
            return 2

        # Any other mosquito species
        if species and species != 'unknown':
            return 2

        return None

    def process_esc50_audios(self):
        """Process ALL ESC-50 environmental sound files."""
        self.logger.info("=" * 70)
        self.logger.info("Processing ALL ESC-50 Categories")
        self.logger.info("=" * 70)

        if not self.esc50_path.exists():
            self.logger.error(f"ESC-50 folder not found: {self.esc50_path}")
            return

        # Get ALL category folders
        category_folders = [f for f in self.esc50_path.iterdir() if f.is_dir()]

        self.logger.info(f"Found {len(category_folders)} ESC-50 categories")

        class_folder = self.output_path / '3'

        for category_folder in tqdm(category_folders, desc="Processing ESC-50 categories"):
            category_name = category_folder.name

            # Get sounds folder
            sounds_folder = category_folder / 'sounds'

            if not sounds_folder.exists():
                self.logger.warning(f"Sounds folder not found for: {category_name}")
                continue

            # Get all WAV files
            audio_files = list(sounds_folder.glob('*.wav'))

            self.logger.info(f"Processing {len(audio_files)} files from {category_name}")

            # Copy files
            for audio_file in audio_files:
                destination = class_folder / audio_file.name
                source_info = f"ESC-50/{category_name}"
                self.copy_audio_file(audio_file, destination, 3, source_info)

    def generate_dataset_report(self):
        """Generate comprehensive dataset statistics report."""
        self.logger.info("=" * 70)
        self.logger.info("Generating Dataset Report")
        self.logger.info("=" * 70)

        # Calculate totals
        total_files = sum(self.stats[f'class_{i}']['count'] for i in range(4))

        # Prepare summary
        summary = {
            'dataset_info': {
                'creation_time': datetime.now().isoformat(),
                'total_files': total_files,
                'duplicates_skipped': self.stats['duplicates'],
                'errors': self.stats['errors']
            },
            'class_distribution': {
                'class_0_aedes_aegypti_male': {
                    'count': self.stats['class_0']['count'],
                    'percentage': (self.stats['class_0']['count'] / total_files * 100) if total_files > 0 else 0,
                    'description': 'Aedes aegypti Male'
                },
                'class_1_aedes_aegypti_female': {
                    'count': self.stats['class_1']['count'],
                    'percentage': (self.stats['class_1']['count'] / total_files * 100) if total_files > 0 else 0,
                    'description': 'Aedes aegypti Female'
                },
                'class_2_other_mosquitoes': {
                    'count': self.stats['class_2']['count'],
                    'percentage': (self.stats['class_2']['count'] / total_files * 100) if total_files > 0 else 0,
                    'description': 'Aedes albopictus + Other species'
                },
                'class_3_environmental': {
                    'count': self.stats['class_3']['count'],
                    'percentage': (self.stats['class_3']['count'] / total_files * 100) if total_files > 0 else 0,
                    'description': 'Environmental sounds (ALL ESC-50 categories)',
                    'categories': self.stats['class_3']['categories']
                }
            }
        }

        # Save JSON report
        json_output = self.output_path / 'dataset_report.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON report saved to: {json_output}")

        # Save text report
        text_output = self.output_path / 'dataset_report.txt'
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MOSQUITO CLASSIFICATION DATASET - REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Creation Time: {summary['dataset_info']['creation_time']}\n")
            f.write(f"Total Audio Files: {summary['dataset_info']['total_files']}\n")
            f.write(f"Duplicates Skipped: {summary['dataset_info']['duplicates_skipped']}\n")
            f.write(f"Errors: {summary['dataset_info']['errors']}\n\n")

            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 80 + "\n")

            for class_key, class_data in summary['class_distribution'].items():
                f.write(f"\n{class_key.upper()}:\n")
                f.write(f"  Description: {class_data['description']}\n")
                f.write(f"  Count: {class_data['count']} files\n")
                f.write(f"  Percentage: {class_data['percentage']:.2f}%\n")

            # Add ESC-50 category breakdown for class 3
            if 'categories' in summary['class_distribution']['class_3_environmental']:
                f.write("\n\nESC-50 CATEGORIES IN CLASS 3\n")
                f.write("-" * 80 + "\n")
                categories = summary['class_distribution']['class_3_environmental']['categories']
                for i, (category, count) in enumerate(sorted(categories.items()), 1):
                    f.write(f"{i:2d}. {category}: {count} files\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"Text report saved to: {text_output}")

        # Print summary to console
        self.logger.info("=" * 70)
        self.logger.info("Dataset Summary")
        self.logger.info("=" * 70)
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"  Class 0 (Aedes aegypti Male): {self.stats['class_0']['count']}")
        self.logger.info(f"  Class 1 (Aedes aegypti Female): {self.stats['class_1']['count']}")
        self.logger.info(f"  Class 2 (Other mosquitoes): {self.stats['class_2']['count']}")
        self.logger.info(f"  Class 3 (Environmental - ALL ESC-50): {self.stats['class_3']['count']}")
        self.logger.info(f"    ESC-50 Categories: {len(self.stats['class_3']['categories'])}")
        self.logger.info(f"Duplicates skipped: {self.stats['duplicates']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info("=" * 70)

    def organize_dataset(self):
        """Main function to organize complete dataset."""
        try:
            # Create folder structure
            self.create_dataset_structure()

            # Process mosquito audios
            self.process_mosquito_audios()

            # Process ESC-50 audios (ALL categories)
            self.process_esc50_audios()

            # Generate report
            self.generate_dataset_report()

            self.logger.info("Dataset organization complete!")
            self.logger.info(f"Dataset saved to: {self.output_path}")

        except Exception as e:
            self.logger.error(f"Fatal error during organization: {e}")
            raise


def main():
    """Main entry point."""
    print("=" * 70)
    print("Dataset Organizer for Mosquito Classification")
    print("=" * 70)
    print()
    print("This script organizes processed audio files into:")
    print("  Class 0: Aedes aegypti Male")
    print("  Class 1: Aedes aegypti Female")
    print("  Class 2: Aedes albopictus + Other species")
    print("  Class 3: Environmental sounds (ALL ESC-50 categories)")
    print()

    # Create organizer
    organizer = DatasetOrganizer(
        mosquito_path=MOSQUITO_INPUT,
        esc50_path=ESC50_INPUT,
        output_path=DATASET_OUTPUT
    )

    # Organize dataset
    organizer.organize_dataset()

    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Dataset location: {DATASET_OUTPUT}/")
    print("Generated files:")
    print(f"  - {DATASET_OUTPUT}/dataset_report.json")
    print(f"  - {DATASET_OUTPUT}/dataset_report.txt")
    print()
    print("Folder structure:")
    print(f"  {DATASET_OUTPUT}/")
    print("    ├── 0/  (Aedes aegypti Male)")
    print("    ├── 1/  (Aedes aegypti Female)")
    print("    ├── 2/  (Aedes albopictus + Other)")
    print("    └── 3/  (ALL ESC-50 Environmental sounds)")


if __name__ == "__main__":
    main()