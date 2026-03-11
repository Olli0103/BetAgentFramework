"""Database engine and session factory.

Uses PostgreSQL via DATABASE_URL environment variable.
Falls back to SQLite for testing if DATABASE_URL is not set.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import Base

load_dotenv()

_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///betagent.db")

engine = create_engine(_DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables defined in models.py."""
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all tables. Use only in tests."""
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
