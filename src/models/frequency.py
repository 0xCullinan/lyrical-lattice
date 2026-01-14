"""
File: src/models/frequency.py
Purpose: SQLAlchemy ORM model for request logs per Section 7.1
"""

from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, Integer, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, INET
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base


class RequestLog(Base):
    """API request log for monitoring and debugging.
    
    Records each API request with timing and status information.
    Does not store sensitive data like audio content.
    
    Attributes:
        id: Primary key.
        request_id: Unique request identifier (UUID).
        timestamp: When the request was received.
        ip_address: Client IP address (hashed in production per RULE-027).
        endpoint: API endpoint path.
        method: HTTP method.
        status_code: HTTP response status.
        response_time_ms: Response latency in milliseconds.
        user_agent: Client user agent string.
        error_message: Error details if request failed.
    """
    
    __tablename__ = "request_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid4,
        nullable=False,
        unique=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    ip_address: Mapped[str] = mapped_column(String(64), nullable=False)  # Hashed IP
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_logs_timestamp", "timestamp", postgresql_ops={"timestamp": "DESC"}),
        Index("idx_logs_ip", "ip_address"),
        Index("idx_logs_endpoint", "endpoint"),
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RequestLog(id={self.id}, endpoint='{self.endpoint}', "
            f"status={self.status_code}, time_ms={self.response_time_ms})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses.
        
        Returns:
            Dictionary representation.
        """
        return {
            "request_id": str(self.request_id),
            "timestamp": self.timestamp.isoformat() + "Z",
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
        }
