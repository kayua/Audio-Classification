import requests
import zipfile
import os
import logging
from pathlib import Path
from typing import Optional


class DownloaderMosquitoesDataset:
    """
    A professional dataset downloader that handles downloading and extracting
    zip files with proper logging and error handling.
    """

    def __init__(
            self,
            url: str,
            output_folder: str = "Datasets",
            chunk_size: int = 8192,
            remove_zip: bool = True,
            log_level: int = logging.INFO
    ):
        """
        Initialize the DatasetDownloader.

        Args:
            url: URL of the dataset to download
            output_folder: Destination folder for extracted files
            chunk_size: Size of chunks for streaming download (bytes)
            remove_zip: Whether to remove zip file after extraction
            log_level: Logging level (default: INFO)
        """
        self.url = url
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        self.remove_zip = remove_zip

        # Setup logging
        self._setup_logging(log_level)

        # Extract filename from URL
        self.zip_filename = os.path.basename(url)

    def _setup_logging(self, log_level: int) -> None:
        """Configure logging for the downloader."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory ready: '{self.output_folder}'")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

    def _download_file(self) -> bool:
        """
        Download the file from the URL.

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            self.logger.info(f"Starting download from: {self.url}")

            response = requests.get(self.url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(self.zip_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)

                        # Log progress every 10%
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if int(progress) % 10 == 0:
                                self.logger.debug(f"Download progress: {progress:.1f}%")

            self.logger.info(f"Download completed: '{self.zip_filename}' ({downloaded_size} bytes)")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            return False

    def _extract_zip(self) -> bool:
        """
        Extract the downloaded zip file.

        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            self.logger.info(f"Extracting '{self.zip_filename}' to '{self.output_folder}'...")

            with zipfile.ZipFile(self.zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.output_folder)
                file_count = len(zip_ref.namelist())

            self.logger.info(f"Extraction completed: {file_count} files extracted")
            return True

        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False

    def _cleanup(self) -> None:
        """Remove the zip file if configured to do so."""
        if self.remove_zip:
            try:
                if os.path.exists(self.zip_filename):
                    os.remove(self.zip_filename)
                    self.logger.info(f"Cleaned up zip file: '{self.zip_filename}'")
            except Exception as e:
                self.logger.warning(f"Failed to remove zip file: {e}")

    def download_and_extract(self) -> bool:
        """
        Main method to download and extract the dataset.

        Returns:
            bool: True if all operations successful, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting dataset download and extraction process")
        self.logger.info("=" * 60)

        try:
            # Create output directory
            self._create_output_directory()

            # Download file
            if not self._download_file():
                self.logger.error("Process aborted: Download failed")
                return False

            # Extract zip
            if not self._extract_zip():
                self.logger.error("Process aborted: Extraction failed")
                return False

            # Cleanup
            self._cleanup()

            self.logger.info("=" * 60)
            self.logger.info("Dataset successfully downloaded and extracted!")
            self.logger.info(f"Location: '{os.path.abspath(self.output_folder)}'")
            self.logger.info("=" * 60)

            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in main process: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Dataset configuration
    DATASET_URL = "https://www.inf.ufrgs.br/aedes-vigilance/resources/mosquito-recordings/Mosquitoes_Dataset.zip"
    OUTPUT_FOLDER = "Datasets"

    # Create downloader instance
    downloader = DownloaderMosquitoesDataset(
        url=DATASET_URL,
        output_folder=OUTPUT_FOLDER,
        chunk_size=8192,
        remove_zip=True,
        log_level=logging.INFO
    )

    # Execute download and extraction
    success = downloader.download_and_extract()

    if success:
        print("\n✓ Process completed successfully!")
    else:
        print("\n✗ Process failed. Check logs for details.")