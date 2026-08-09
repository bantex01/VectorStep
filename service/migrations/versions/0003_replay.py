"""replay shadow eval

Adds replay/shadow evaluation support (SPEC-replay-shadow-eval.md): a
`replay_of` JSON descriptor column on pipeline_runs, set only for the
synthetic runs a replay batch creates so the report can reconstruct what was
replayed. NULL for every ordinary run.

Revision ID: 0003_replay
Revises: 0002_durable_runs
Create Date: 2026-08-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003_replay'
down_revision: Union[str, Sequence[str], None] = '0002_durable_runs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('pipeline_runs', sa.Column('replay_of', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('pipeline_runs', 'replay_of')
