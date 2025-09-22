from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import datetime

DATABASE_URL = "sqlite:///./results.db"  # file stored locally

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    label = Column(String)
    confidence = Column(Float)
    explainability = Column(String, nullable=True)  # path to heatmap if you add later
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the DB tables
Base.metadata.create_all(bind=engine)
