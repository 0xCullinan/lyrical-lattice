"""
File: src/services/file_service.py
Purpose: File upload handling and cleanup per REQ-SEC-001, REQ-SEC-002, REQ-SEC-003
"""

import os
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, BinaryIO
import aiofiles
import aiofiles.os

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.validators import validate_audio_file, ValidationError

logger = get_logger(__name__)


class FileService:
    """File upload and cleanup service.
    
    Handles audio file uploads with validation and automatic cleanup.
    Files are deleted within 24 hours per REQ-SEC-003.
    
    Attributes:
        upload_dir: Directory for temporary uploads.
        max_file_size: Maximum file size in bytes.
    """
    
    ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}
    CLEANUP_INTERVAL_HOURS = 1  # Run cleanup hourly
    MAX_FILE_AGE_HOURS = 24  # Delete files older than 24 hours
    
    def __init__(self):
        """Initialize file service."""
        self.upload_dir = Path(settings.upload_dir)
        self.max_file_size = settings.max_file_size_bytes
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize file service and create upload directory."""
        logger.info(f"Initializing file service, upload dir: {self.upload_dir}")
        
        # Create upload directory if it doesn't exist
        await aiofiles.os.makedirs(self.upload_dir, exist_ok=True)
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("File service initialized")
    
    async def close(self) -> None:
        """Close file service and cancel cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("File service closed")
    
    async def save_upload(
        self,
        file_content: bytes,
        filename: str,
    ) -> Path:
        """Save an uploaded file with validation.
        
        Args:
            file_content: Raw file bytes.
            filename: Original filename.
            
        Returns:
            Path to saved file.
            
        Raises:
            ValidationError: If file is invalid or too large.
        """
        # Check file size (REQ-S2P-002)
        if len(file_content) > self.max_file_size:
            raise ValidationError(
                f"File size exceeds {settings.max_file_size_mb} MB limit",
                field="file",
                value=filename,
            )
        
        # Get file extension
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}",
                field="file",
                value=filename,
            )
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:16]
        safe_filename = f"{unique_id}.{ext}"
        file_path = self.upload_dir / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        
        # Validate magic bytes match extension (REQ-SEC-001, REQ-SEC-002)
        try:
            detected_format = validate_audio_file(file_path, ext)
            logger.debug(f"Saved upload: {safe_filename}, format: {detected_format}")
        except ValidationError:
            # Delete file if validation fails
            await self.delete_file(file_path)
            raise
        
        return file_path
    
    async def save_upload_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        chunk_size: int = 65536,
    ) -> Path:
        """Save an uploaded file from a stream.
        
        Args:
            file_stream: File-like object to read from.
            filename: Original filename.
            chunk_size: Size of chunks to read.
            
        Returns:
            Path to saved file.
            
        Raises:
            ValidationError: If file is invalid or too large.
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}",
                field="file",
                value=filename,
            )
        
        unique_id = uuid.uuid4().hex[:16]
        safe_filename = f"{unique_id}.{ext}"
        file_path = self.upload_dir / safe_filename
        
        total_size = 0
        
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = file_stream.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                if total_size > self.max_file_size:
                    await f.close()
                    await self.delete_file(file_path)
                    raise ValidationError(
                        f"File size exceeds {settings.max_file_size_mb} MB limit",
                        field="file",
                        value=filename,
                    )
                
                await f.write(chunk)
        
        # Validate magic bytes
        try:
            validate_audio_file(file_path, ext)
        except ValidationError:
            await self.delete_file(file_path)
            raise
        
        return file_path
    
    async def delete_file(self, file_path: Path) -> None:
        """Delete a file.
        
        Args:
            file_path: Path to file to delete.
        """
        try:
            if file_path.exists():
                await aiofiles.os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
        except OSError as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
    
    async def delete_after_processing(self, file_path: Path) -> None:
        """Delete a file after processing is complete.
        
        This should be called after audio processing to immediately
        remove the file rather than waiting for cleanup.
        
        Args:
            file_path: Path to file to delete.
        """
        await self.delete_file(file_path)
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up old files per REQ-SEC-003."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_HOURS * 3600)
                await self._cleanup_old_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_old_files(self) -> None:
        """Delete files older than MAX_FILE_AGE_HOURS."""
        cutoff = datetime.now() - timedelta(hours=self.MAX_FILE_AGE_HOURS)
        deleted_count = 0
        
        try:
            for entry in os.scandir(self.upload_dir):
                if entry.is_file():
                    mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                    if mtime < cutoff:
                        await aiofiles.os.remove(entry.path)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files")
        except Exception as e:
            logger.error(f"File cleanup error: {e}")
    
    async def get_file_info(self, file_path: Path) -> Optional[dict]:
        """Get information about a file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            File info dict or None if file doesn't exist.
        """
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }


# Global file service instance
file_service = FileService()
