"""
File: alembic/versions/20251216_0001_initial_schema.py
Purpose: Initial database schema migration
"""

"""Initial schema

Revision ID: 0001
Revises: 
Create Date: 2025-12-16 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables."""
    
    # Create words table
    op.create_table(
        'words',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('word', sa.String(length=100), nullable=False),
        sa.Column('ipa', sa.String(length=200), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=False, server_default='en_US'),
        sa.Column('frequency', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('vector_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('word', 'ipa', 'language', name='uq_word_ipa_language')
    )
    
    # Create indexes for words table
    op.create_index('idx_words_word', 'words', ['word'])
    op.create_index('idx_words_ipa', 'words', ['ipa'])
    op.create_index('idx_words_language', 'words', ['language'])
    op.create_index('idx_words_frequency', 'words', [sa.text('frequency DESC')])
    op.create_index('idx_words_vector_id', 'words', ['vector_id'])
    
    # Create phonemes table
    op.create_table(
        'phonemes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('ipa_category', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('example_word', sa.Text(), nullable=True),
        sa.Column('example_ipa', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol')
    )
    
    # Create request_logs table
    op.create_table(
        'request_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('ip_address', sa.String(length=64), nullable=False),
        sa.Column('endpoint', sa.String(length=100), nullable=False),
        sa.Column('method', sa.String(length=10), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('response_time_ms', sa.Integer(), nullable=False),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_id')
    )
    
    # Create indexes for request_logs table
    op.create_index('idx_logs_timestamp', 'request_logs', [sa.text('timestamp DESC')])
    op.create_index('idx_logs_ip', 'request_logs', ['ip_address'])
    op.create_index('idx_logs_endpoint', 'request_logs', ['endpoint'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('request_logs')
    op.drop_table('phonemes')
    op.drop_table('words')
