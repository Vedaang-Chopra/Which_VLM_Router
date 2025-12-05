"""Database connection utilities."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from ares.configs.db_config import DB_URL


_engine = None


def get_engine():
    """Get or create SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
    return _engine


def get_session():
    """Create a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection():
    """Test database connection."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT NOW()"))
            row = result.fetchone()
            print(f"✓ Connected to database. Server time: {row[0]}")
            return True
    except OperationalError as e:
        print(f"✗ Failed to connect: {e}")
        return False


def execute_sql(sql: str):
    """Execute raw SQL statement."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql))