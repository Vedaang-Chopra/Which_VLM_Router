from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

_engine = None
_session_factory = None

def get_db_engine(db_url: str):
    """
    Get or create a singleton SQLAlchemy engine.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(db_url, pool_size=10, max_overflow=20)
    return _engine

def get_session_factory(db_url: str):
    """
    Get a scoped session factory.
    """
    global _session_factory
    if _session_factory is None:
        engine = get_db_engine(db_url)
        _session_factory = scoped_session(sessionmaker(bind=engine))
    return _session_factory
