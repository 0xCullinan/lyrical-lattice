"""
File: src/models/base.py
Purpose: SQLAlchemy base model and common mixins
"""

from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.
    
    All models should inherit from this class.
    """
    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns.
    
    Automatically sets created_at on insert and updates
    updated_at on every update.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
