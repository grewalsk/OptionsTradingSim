# models.py

from sqlalchemy import Column, Integer, String, DECIMAL, Date, ForeignKey, TIMESTAMP, JSON
from sqlalchemy.orm import relationship
from database import Base

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, index=True)
    ticker_symbol = Column(String(10), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    open_price = Column(DECIMAL(15, 4))
    high_price = Column(DECIMAL(15, 4))
    low_price = Column(DECIMAL(15, 4))
    close_price = Column(DECIMAL(15, 4))
    volume = Column(Integer)
    
    __table_args__ = (
        # Ensure uniqueness on ticker_symbol and timestamp
        {'postgresql_conflict_target': ('ticker_symbol', 'timestamp'), 'postgresql_on_conflict': 'DO NOTHING'},
    )

class Volatility(Base):
    __tablename__ = 'volatility'

    id = Column(Integer, primary_key=True, index=True)
    ticker_symbol = Column(String(10), ForeignKey('market_data.ticker_symbol'), nullable=False)
    date = Column(Date, nullable=False)
    volatility = Column(DECIMAL(10, 4))

class OptionContract(Base):
    __tablename__ = 'option_contracts'

    id = Column(Integer, primary_key=True, index=True)
    contract_id = Column(String(50), unique=True, nullable=False)
    ticker_symbol = Column(String(10), ForeignKey('market_data.ticker_symbol'), nullable=False)
    option_type = Column(String(4), nullable=False)
    strike_price = Column(DECIMAL(15, 4), nullable=False)
    expiration_date = Column(Date, nullable=False)
    underlying_asset_price = Column(DECIMAL(15, 4))

class ModelConfiguration(Base):
    __tablename__ = 'model_configurations'

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)
    parameters = Column(JSON)
    training_date = Column(TIMESTAMP)

    # Relationship to ModelResults
    results = relationship("ModelResult", back_populates="model_configuration")

class ModelResult(Base):
    __tablename__ = 'model_results'

    id = Column(Integer, primary_key=True, index=True)
    model_configuration_id = Column(Integer, ForeignKey('model_configurations.id'), nullable=False)
    option_contract_id = Column(Integer, ForeignKey('option_contracts.id'), nullable=False)
    predicted_price = Column(DECIMAL(15, 4))
    actual_price = Column(DECIMAL(15, 4))
    error_metric = Column(DECIMAL(10, 4))

    # Relationships
    model_configuration = relationship("ModelConfiguration", back_populates="results")
    option_contract = relationship("OptionContract")

class MarketSentiment(Base):
    __tablename__ = 'market_sentiment'

    id = Column(Integer, primary_key=True, index=True)
    ticker_symbol = Column(String(10), ForeignKey('market_data.ticker_symbol'), nullable=False)
    sentiment_date = Column(Date, nullable=False)
    sentiment_score = Column(DECIMAL(5, 2))
    source = Column(String(100))

class PerformanceMetric(Base):
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(DECIMAL(15, 6))
    recorded_at = Column(TIMESTAMP)
