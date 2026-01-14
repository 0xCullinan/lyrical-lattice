"""
File: src/services/db_service.py
Purpose: Async PostgreSQL database service with connection pooling per REQ-PERF-004
"""

from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy import text, select, func
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import settings
from src.utils.logger import get_logger
from src.models.base import Base
from src.models.word import Word

logger = get_logger(__name__)


class DatabaseService:
    """Async PostgreSQL database service.
    
    Provides connection pooling, session management, and common
    database operations for the application.
    
    Attributes:
        engine: SQLAlchemy async engine.
        session_factory: Async session factory.
    """
    
    def __init__(self):
        """Initialize database service."""
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory.
        
        Creates connection pool with min 5, max 20 connections per REQ-PERF-004.
        """
        logger.info("Initializing database connection pool")
        
        self._engine = create_async_engine(
            settings.database_url,
            pool_size=5,        # Min connections per REQ-PERF-004
            max_overflow=15,    # Max 20 total (5 + 15)
            pool_pre_ping=True, # Check connection health
            pool_recycle=3600,  # Recycle connections hourly
            echo=settings.log_level == "DEBUG",
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        logger.info("Database connection pool initialized")
    
    async def close(self) -> None:
        """Close database connections gracefully."""
        if self._engine:
            logger.info("Closing database connection pool")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session.
        
        Yields:
            AsyncSession: Database session that auto-commits on success
            and rolls back on error.
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            await session.close()
    
    async def check_health(self) -> tuple[bool, Optional[int]]:
        """Check database health.
        
        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        import time
        
        if not self._engine:
            return False, None
        
        try:
            start = time.perf_counter()
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency_ms = int((time.perf_counter() - start) * 1000)
            return True, latency_ms
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False, None
    
    async def create_tables(self) -> None:
        """Create all database tables.
        
        For development/testing only. Use Alembic migrations in production.
        """
        if not self._engine:
            raise RuntimeError("Database not initialized")
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def get_corpus_size(self) -> int:
        """Get the total number of words in the corpus.
        
        Returns:
            Number of word entries.
        """
        async with self.session() as session:
            result = await session.execute(select(func.count(Word.id)))
            return result.scalar() or 0
    
    async def get_word_by_text(self, word: str, language: str = "en_US") -> Optional[Word]:
        """Get word entry by text.
        
        Args:
            word: Word to look up.
            language: Language code.
            
        Returns:
            Word entry if found, None otherwise.
        """
        async with self.session() as session:
            result = await session.execute(
                select(Word).where(
                    Word.word == word,
                    Word.language == language,
                ).limit(1)
            )
            return result.scalar_one_or_none()
    
    async def get_words_by_vector_ids(self, vector_ids: list[int]) -> list[Word]:
        """Get words by their FAISS vector IDs.
        
        Args:
            vector_ids: List of vector IDs to look up.
            
        Returns:
            List of Word entries.
        """
        async with self.session() as session:
            result = await session.execute(
                select(Word).where(Word.vector_id.in_(vector_ids))
            )
            return list(result.scalars().all())
    
    async def insert_word(self, word_data: dict[str, Any]) -> Word:
        """Insert a new word entry.
        
        Args:
            word_data: Word data dictionary.
            
        Returns:
            Created Word entry.
        """
        async with self.session() as session:
            word = Word(**word_data)
            session.add(word)
            await session.flush()
            return word
    
    async def bulk_insert_words(self, words_data: list[dict[str, Any]]) -> int:
        """Bulk insert word entries with conflict handling.
        
        Uses ON CONFLICT DO NOTHING to skip duplicates.
        
        Args:
            words_data: List of word data dictionaries.
            
        Returns:
            Number of inserted entries (approximate, usually returns total batch size even if ignored).
        """
        if not words_data:
            return 0
            
        from sqlalchemy.dialects.postgresql import insert
        
        async with self.session() as session:
            stmt = insert(Word).values(words_data)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["word", "ipa", "language"]
            )
            result = await session.execute(stmt)
            # rowcount for DO NOTHING is not always reliable in all drivers, but for asyncpg it implies affected rows
            return result.rowcount


# Global database service instance
db_service = DatabaseService()
