# database.py (updated)
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import datetime

DATABASE_URL = "sqlite:///./results.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ScanResult(Base):
    __tablename__ = "scan_results"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    label = Column(String)
    confidence = Column(Float)
    model_version = Column(String)
    explainability = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Create all tables
Base.metadata.create_all(bind=engine)
