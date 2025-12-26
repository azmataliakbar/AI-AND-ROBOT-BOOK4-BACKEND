from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class DatabaseSettings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost/dbname"
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False
    )


db_settings = DatabaseSettings()

# Create engine
engine = create_engine(
    db_settings.database_url,
    pool_pre_ping=True,  # Verify connections before using them
    pool_size=5,
    max_overflow=10
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Test connection and print status
def test_database_connection():
    """Test database connection and print status"""
    try:
        # Try to connect
        connection = engine.connect()
        connection.close()
        print("SUCCESS: Neon PostgreSQL connected successfully!")
        return True
    except Exception as e:
        print(f"WARN: Database connection failed: {str(e)[:100]}")
        print("   Backend will continue with limited functionality")
        return False


# Test connection on import
test_database_connection()