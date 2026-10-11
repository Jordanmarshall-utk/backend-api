"""Phase-mapper libraries and runs."""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "phase_mapper_libraries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("composition_profile", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "phase_mapper_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("library_id", sa.BigInteger(), sa.ForeignKey("phase_mapper_libraries.id"), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("dir_path", sa.Text(), nullable=False),
        sa.Column("job_pid", sa.BigInteger(), nullable=True),
        sa.Column("return_code", sa.BigInteger(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("frozen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("phase_mapper_runs")
    op.drop_table("phase_mapper_libraries")
