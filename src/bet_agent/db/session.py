"""Database engine and session factory.

Uses PostgreSQL via DATABASE_URL environment variable.
Fails hard if DATABASE_URL is not set (no silent SQLite fallback in production).
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import Base

_DATABASE_URL = os.getenv("DATABASE_URL")

if _DATABASE_URL is None:
    if os.getenv("BETAGENT_ENV", "").lower() == "test":
        _DATABASE_URL = "sqlite:///betagent_test.db"
    else:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Set BETAGENT_ENV=test to use SQLite for testing."
        )

engine = create_engine(_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables defined in models.py."""
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all tables. Only works when BETAGENT_ENV=test."""
    if os.getenv("BETAGENT_ENV", "").lower() != "test":
        raise RuntimeError("drop_db() is only allowed when BETAGENT_ENV=test")
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_session():
    """Provide a transactional session scope."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
