-- Driver Behavior Analytics Database Schema
-- Initialize PostgreSQL database for production deployment

-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS driver_analytics;

-- Connect to the database
\c driver_analytics;

-- Create schemas for data organization
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS model_results;
CREATE SCHEMA IF NOT EXISTS audit;

-- Driver profiles table
CREATE TABLE IF NOT EXISTS raw_data.driver_profiles (
    driver_id VARCHAR(50) PRIMARY KEY,
    age INTEGER CHECK (age >= 16 AND age <= 100),
    experience_years DECIMAL(5,2) CHECK (experience_years >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Driver behavior metrics table
CREATE TABLE IF NOT EXISTS raw_data.driver_metrics (
    id SERIAL PRIMARY KEY,
    driver_id VARCHAR(50) REFERENCES raw_data.driver_profiles(driver_id),
    measurement_date DATE NOT NULL,
    speed_variance DECIMAL(8,3) CHECK (speed_variance >= 0),
    harsh_acceleration_events INTEGER CHECK (harsh_acceleration_events >= 0),
    harsh_braking_events INTEGER CHECK (harsh_braking_events >= 0),
    night_driving_hours DECIMAL(6,2) CHECK (night_driving_hours >= 0),
    weekend_driving_ratio DECIMAL(3,2) CHECK (weekend_driving_ratio >= 0 AND weekend_driving_ratio <= 1),
    avg_trip_distance DECIMAL(8,2) CHECK (avg_trip_distance >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Survival analysis data table
CREATE TABLE IF NOT EXISTS processed_data.survival_data (
    id SERIAL PRIMARY KEY,
    driver_id VARCHAR(50) REFERENCES raw_data.driver_profiles(driver_id),
    duration_days INTEGER CHECK (duration_days > 0),
    event_occurred BOOLEAN NOT NULL,
    analysis_date DATE NOT NULL,
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk scores table
CREATE TABLE IF NOT EXISTS model_results.risk_scores (
    id SERIAL PRIMARY KEY,
    driver_id VARCHAR(50) REFERENCES raw_data.driver_profiles(driver_id),
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_category VARCHAR(20) CHECK (risk_category IN ('low', 'medium', 'high', 'critical')),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    model_version VARCHAR(20) NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_importance JSONB,
    recommendations TEXT[]
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_results.model_performance (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    c_index DECIMAL(5,4),
    aic DECIMAL(10,3),
    accuracy DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_samples INTEGER,
    validation_metrics JSONB
);

-- API audit log
CREATE TABLE IF NOT EXISTS audit.api_requests (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(50),
    request_body JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_driver_metrics_driver_id ON raw_data.driver_metrics(driver_id);
CREATE INDEX IF NOT EXISTS idx_driver_metrics_date ON raw_data.driver_metrics(measurement_date);
CREATE INDEX IF NOT EXISTS idx_survival_data_driver_id ON processed_data.survival_data(driver_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_driver_id ON model_results.risk_scores(driver_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_date ON model_results.risk_scores(prediction_date);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON audit.api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON audit.api_requests(created_at);

-- Create views for common queries
CREATE OR REPLACE VIEW processed_data.driver_risk_summary AS
SELECT 
    dp.driver_id,
    dp.age,
    dp.experience_years,
    rs.risk_score,
    rs.risk_category,
    rs.confidence_score,
    rs.prediction_date
FROM raw_data.driver_profiles dp
LEFT JOIN LATERAL (
    SELECT risk_score, risk_category, confidence_score, prediction_date
    FROM model_results.risk_scores 
    WHERE driver_id = dp.driver_id 
    ORDER BY prediction_date DESC 
    LIMIT 1
) rs ON true;

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_driver_profiles_updated_at 
    BEFORE UPDATE ON raw_data.driver_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA raw_data TO analytics_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA processed_data TO analytics_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA model_results TO analytics_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO analytics_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA raw_data TO analytics_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA processed_data TO analytics_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA model_results TO analytics_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA audit TO analytics_user;

-- Insert sample data for testing (optional)
INSERT INTO raw_data.driver_profiles (driver_id, age, experience_years) VALUES
('D001', 28, 5.2),
('D002', 35, 12.8),
('D003', 42, 18.5)
ON CONFLICT (driver_id) DO NOTHING;

-- Success message
DO $
BEGIN
    RAISE NOTICE 'Driver Behavior Analytics database initialized successfully!';
END $;
