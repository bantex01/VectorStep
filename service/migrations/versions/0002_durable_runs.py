"""durable runs

Adds resume-after-restart support (SPEC-durable-runs.md): a config_fingerprint +
resumed_at on pipeline_runs, and a pending_approvals table that durably mirrors
executors/human.py's in-memory HITL wait state so an approval request survives a
restart.

Revision ID: 0002_durable_runs
Revises: 0001_baseline
Create Date: 2026-08-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_durable_runs'
down_revision: Union[str, Sequence[str], None] = '0001_baseline'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('pipeline_runs', sa.Column('config_fingerprint', sa.String(), nullable=True))
    op.add_column('pipeline_runs', sa.Column('resumed_at', sa.DateTime(), nullable=True))
    op.create_table('pending_approvals',
    sa.Column('token', sa.String(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('step_name', sa.String(), nullable=False),
    sa.Column('pipeline_name', sa.String(), nullable=True),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('team', sa.String(), nullable=True),
    sa.Column('stage', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('token')
    )
    op.create_index(op.f('ix_pending_approvals_run_id'), 'pending_approvals', ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_pending_approvals_run_id'), table_name='pending_approvals')
    op.drop_table('pending_approvals')
    op.drop_column('pipeline_runs', 'resumed_at')
    op.drop_column('pipeline_runs', 'config_fingerprint')
