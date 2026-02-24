"""phase6: conversations table, document summary, guardrail_blocked enum

Revision ID: c8a2f6d91e03
Revises: b19477e427c6
Create Date: 2026-02-24 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8a2f6d91e03"
down_revision: str | None = "b19477e427c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add GUARDRAIL_BLOCKED to auditaction enum
    op.execute("ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'GUARDRAIL_BLOCKED'")

    # 2. Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("client_id", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_client_id"), "conversations", ["client_id"], unique=False)

    # 3. Add summary column to documents
    op.add_column("documents", sa.Column("summary", sa.Text(), nullable=True))

    # 4. Add conversation_id FK to queries
    op.add_column("queries", sa.Column("conversation_id", sa.Uuid(), nullable=True))
    op.create_index(op.f("ix_queries_conversation_id"), "queries", ["conversation_id"], unique=False)
    op.create_foreign_key(
        "fk_queries_conversation_id",
        "queries",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # 4. Drop conversation_id FK from queries
    op.drop_constraint("fk_queries_conversation_id", "queries", type_="foreignkey")
    op.drop_index(op.f("ix_queries_conversation_id"), table_name="queries")
    op.drop_column("queries", "conversation_id")

    # 3. Drop summary column from documents
    op.drop_column("documents", "summary")

    # 2. Drop conversations table
    op.drop_index(op.f("ix_conversations_client_id"), table_name="conversations")
    op.drop_table("conversations")

    # 1. Cannot remove enum value in PostgreSQL (would need to recreate type)
