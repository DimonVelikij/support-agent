"""create_request_table

Revision ID: a9f35b21adf9
Revises: 
Create Date: 2025-11-09 22:42:13.369317

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a9f35b21adf9'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE requests (
            id SERIAL NOT NULL,
            query TEXT NOT NULL,
            qdrant_result JSON NOT NULL DEFAULT '{}',
            llm_result JSON NOT NULL DEFAULT '{}',
            feedback BOOL DEFAULT NULL,
            feedback_comment TEXT DEFAULT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (id)
        )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE requests
        """
    )
