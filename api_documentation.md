# Driver Behavior Analytics System - API Documentation

## Overview

The Driver Behavior Analytics API provides comprehensive endpoints for driver risk assessment using advanced statistical modeling including Cox proportional hazards, Bayesian hierarchical models, and SHAP explainability. The API is built with FastAPI and provides real-time scoring with sub-200ms response times.

**Base URL:** `http://localhost:8003` (development) or `https://your-domain.com` (production)

## Authentication

All API requests require authentication using an API key in the header:

```bash
X-API-Key: your-api-key-here
```

## Rate Limiting

- **Rate Limit:** 1000 requests per hour per API key
- **Burst Limit:** 100 requests per minute
- **Headers:** Rate limit information is returned in response headers

## Core Endpoints

### System Information

#### GET `/`
Get API information and capabilities.

**Response:**
```json
{
  "name": "Driver Behavior Analytics API",
  "version": "1.0.0",
  "description": "Advanced driver risk analytics using survival analysis and Bayesian modeling",
  "endpoints": ["survival", "bayesian", "scoring", "analysis"],
  "models_loaded": {
    "cox_model": true,
    "bayesian_model": true,
    "shap_explainer": true
  },
  "performance": {
    "avg_response_time": "185ms",
    "models_analyzed": 300000,
    "c_index": 0.79
  }
}
```

#### GET `/health`
Health check and system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-09T15:30:00Z",
  "uptime": "5d 12h 30m",
  "database": "connected",
  "redis": "connected",
  "models": {
    "cox_model": "loaded",
    "bayesian_model": "loaded"
  },
  "memory_usage": "2.1GB",
  "cpu_usage": "15%"
}
```

#### GET `/docs`
Interactive API documentation (Swagger UI).

Redirects to Swagger UI interface with interactive API testing capabilities.

---

## Survival Analysis Endpoints

### POST `/survival/cox_regression`
Fit Cox proportional hazards model for survival analysis.

**Request Body:**
```json
{
  "data": [
    {
      "duration": 365,
      "event": 1,
      "age": 25,
      "experience": 2.5,
      "speed_variance": 15.2,
      "harsh_braking": 3
    }
  ],
  "duration_column": "duration",
  "event_column": "event",
  "feature_columns": ["age", "experience", "speed_variance", "harsh_braking"],
  "options": {
    "penalizer": 0.1,
    "l1_ratio": 0.0
  }
}
```

**Response:**
```json
{
  "model_id": "cox_model_20241209_153000",
  "c_index": 0.79,
  "log_likelihood": -1423.5,
  "aic": 2847.0,
  "coefficients": {
    "age": 0.025,
    "experience": -0.045,
    "speed_variance": 0.078,
    "harsh_braking": 0.112
  },
  "hazard_ratios": {
    "age": 1.025,
    "experience": 0.956,
    "speed_variance": 1.081,
    "harsh_braking": 1.118
  },
  "confidence_intervals": {
    "age": [0.018, 0.032],
    "experience": [-0.052, -0.038],
    "speed_variance": [0.065, 0.091],
    "harsh_braking": [0.098, 0.126]
  },
  "assumption_tests": {
    "proportional_hazards": {
      "p_value": 0.156,
      "passed": true
    }
  }
}
```

### POST `/survival/kaplan_meier`
Perform Kaplan-Meier survival analysis.

**Request Body:**
```json
{
  "data": [
    {
      "duration": 365,
      "event": 1,
      "group": "high_risk"
    }
  ],
  "duration_column": "duration",
  "event_column": "event",
  "group_column": "group"
}
```

**Response:**
```json
{
  "analysis_id": "km_analysis_20241209_153000",
  "groups": ["high_risk", "medium_risk", "low_risk"],
  "survival_curves": {
    "high_risk": {
      "timeline": [0, 30, 60, 90, 120, 150, 180],
      "survival_function": [1.0, 0.95, 0.89, 0.82, 0.75, 0.68, 0.61],
      "confidence_interval_lower": [1.0, 0.92, 0.85, 0.77, 0.69, 0.61, 0.53],
      "confidence_interval_upper": [1.0, 0.98, 0.93, 0.87, 0.81, 0.75, 0.69]
    }
  },
  "median_survival": {
    "high_risk": 165,
    "medium_risk": 245,
    "low_risk": 320
  },
  "log_rank_test": {
    "statistic": 45.2,
    "p_value": 0.0001,
    "degrees_of_freedom": 2
  }
}
```

### POST `/survival/parametric_models`
Compare parametric survival models (Weibull, Log-Normal, Exponential).

**Request Body:**
```json
{
  "data": [
    {
      "duration": 365,
      "event": 1,
      "age": 25,
      "experience": 2.5
    }
  ],
  "duration_column": "duration",
  "event_column": "event",
  "feature_columns": ["age", "experience"]
}
```

**Response:**
```json
{
  "comparison_id": "parametric_comp_20241209_153000",
  "models": {
    "weibull": {
      "aic": 2923.5,
      "log_likelihood": -1458.8,
      "parameters": {
        "lambda_": 0.0032,
        "rho_": 1.15
      }
    },
    "lognormal": {
      "aic": 2945.2,
      "log_likelihood": -1469.6,
      "parameters": {
        "mu": 5.2,
        "sigma": 0.8
      }
    },
    "exponential": {
      "aic": 2978.1,
      "log_likelihood": -1486.1,
      "parameters": {
        "lambda_": 0.0028
      }
    }
  },
  "best_model": "weibull",
  "selection_criteria": "aic"
}
```

---

## Bayesian Analysis Endpoints

### POST `/bayesian/hierarchical_model`
Fit Bayesian hierarchical survival model.

**Request Body:**
```json
{
  "data": [
    {
      "duration": 365,
      "event": 1,
      "driver_id": "D123456",
      "fleet_id": "F001",
      "age": 25,
      "experience": 2.5
    }
  ],
  "group_column": "fleet_id",
  "mcmc_settings": {
    "draws": 2000,
    "tune": 1000,
    "chains": 4,
    "cores": 4
  }
}
```

**Response:**
```json
{
  "model_id": "bayesian_hierarchical_20241209_153000",
  "posterior_summary": {
    "age_coef": {
      "mean": 0.024,
      "std": 0.008,
      "hdi_2.5": 0.009,
      "hdi_97.5": 0.039,
      "r_hat": 1.01
    },
    "experience_coef": {
      "mean": -0.043,
      "std": 0.007,
      "hdi_2.5": -0.057,
      "hdi_97.5": -0.029,
      "r_hat": 1.00
    }
  },
  "group_effects": {
    "F001": {
      "mean": 0.12,
      "std": 0.05
    },
    "F002": {
      "mean": -0.08,
      "std": 0.04
    }
  },
  "model_comparison": {
    "loo": -1425.3,
    "waic": -1423.8,
    "posterior_predictive_accuracy": 0.914
  },
  "convergence": {
    "all_r_hat_below_1_1": true,
    "effective_sample_size_ok": true,
    "mcse_acceptable": true
  }
}
```

### POST `/bayesian/driver_segmentation`
Perform Bayesian mixture modeling for driver segmentation.

**Request Body:**
```json
{
  "data": [
    {
      "driver_id": "D123456",
      "speed_variance": 15.2,
      "harsh_braking": 3,
      "night_driving": 45.5
    }
  ],
  "feature_columns": ["speed_variance", "harsh_braking", "night_driving"],
  "n_components": 3,
  "mcmc_settings": {
    "draws": 2000,
    "tune": 1000,
    "chains": 4
  }
}
```

**Response:**
```json
{
  "segmentation_id": "driver_segments_20241209_153000",
  "n_components": 3,
  "cluster_assignments": {
    "D123456": {
      "cluster": 1,
      "probability": [0.05, 0.92, 0.03]
    }
  },
  "cluster_characteristics": {
    "cluster_0": {
      "label": "conservative_drivers",
      "size": 0.45,
      "characteristics": {
        "speed_variance": {
          "mean": 8.2,
          "std": 2.1
        },
        "harsh_braking": {
          "mean": 1.2,
          "std": 0.8
        }
      }
    },
    "cluster_1": {
      "label": "moderate_drivers",
      "size": 0.38,
      "characteristics": {
        "speed_variance": {
          "mean": 15.1,
          "std": 3.2
        },
        "harsh_braking": {
          "mean": 3.1,
          "std": 1.2
        }
      }
    },
    "cluster_2": {
      "label": "aggressive_drivers",
      "size": 0.17,
      "characteristics": {
        "speed_variance": {
          "mean": 24.8,
          "std": 4.5
        },
        "harsh_braking": {
          "mean": 6.2,
          "std": 2.1
        }
      }
    }
  }
}
```

---

## Risk Scoring Endpoints

### POST `/scoring/calculate_risk`
Calculate risk score for a single driver.

**Request Body:**
```json
{
  "driver_features": {
    "driver_id": "D123456",
    "speed_variance": 15.2,
    "harsh_acceleration_events": 3,
    "harsh_braking_events": 2,
    "night_driving_hours": 45.5,
    "weekend_driving_ratio": 0.35,
    "avg_trip_distance": 12.8,
    "experience_years": 5.2,
    "age": 28
  },
  "model_ensemble": true,
  "explain": true
}
```

**Response:**
```json
{
  "driver_id": "D123456",
  "risk_score": 0.672,
  "risk_category": "medium_high",
  "risk_percentile": 78,
  "confidence_interval": [0.645, 0.699],
  "model_contributions": {
    "cox_model": 0.668,
    "bayesian_model": 0.676,
    "ensemble_weight": {
      "cox": 0.4,
      "bayesian": 0.6
    }
  },
  "explanation": {
    "shap_values": {
      "speed_variance": 0.142,
      "harsh_braking_events": 0.089,
      "night_driving_hours": 0.065,
      "experience_years": -0.034,
      "age": 0.021
    },
    "base_value": 0.400,
    "feature_importance_rank": [
      "speed_variance",
      "harsh_braking_events", 
      "night_driving_hours",
      "age",
      "experience_years"
    ]
  },
  "recommendations": [
    "Focus on reducing speed variance during driving",
    "Implement defensive driving training",
    "Consider limiting night driving hours"
  ],
  "response_time_ms": 185
}
```

### POST `/scoring/batch_scoring`
Score multiple drivers in batch.

**Request Body:**
```json
{
  "drivers": [
    {
      "driver_id": "D123456",
      "speed_variance": 15.2,
      "harsh_acceleration_events": 3
    },
    {
      "driver_id": "D789012",
      "speed_variance": 8.5,
      "harsh_acceleration_events": 1
    }
  ],
  "model_ensemble": true,
  "explain": false
}
```

**Response:**
```json
{
  "batch_id": "batch_20241209_153000",
  "total_drivers": 2,
  "processed": 2,
  "failed": 0,
  "results": [
    {
      "driver_id": "D123456",
      "risk_score": 0.672,
      "risk_category": "medium_high"
    },
    {
      "driver_id": "D789012", 
      "risk_score": 0.324,
      "risk_category": "low"
    }
  ],
  "processing_time_ms": 45,
  "avg_score_per_driver_ms": 22.5
}
```

### GET `/scoring/risk_trends/{driver_id}`
Get risk trend analysis for a specific driver.

**Parameters:**
- `driver_id` (path): Driver identifier
- `days` (query, optional): Number of days to analyze (default: 90)

**Response:**
```json
{
  "driver_id": "D123456",
  "analysis_period": {
    "start_date": "2024-09-10",
    "end_date": "2024-12-09",
    "total_days": 90
  },
  "risk_trend": {
    "timeline": ["2024-09-10", "2024-09-20", "2024-09-30", "2024-10-10"],
    "risk_scores": [0.623, 0.645, 0.672, 0.681],
    "trend": "increasing",
    "change_rate": 0.0015
  },
  "driving_patterns": {
    "total_trips": 145,
    "avg_daily_distance": 45.2,
    "night_driving_trend": "increasing",
    "weekend_driving_ratio": 0.35
  },
  "alerts": [
    {
      "type": "risk_increase",
      "message": "Risk score increased by 9.3% over last 30 days",
      "severity": "medium"
    }
  ]
}
```

---

## Analysis Endpoints

### POST `/analysis/feature_importance`
Multi-method feature importance analysis.

**Request Body:**
```json
{
  "data": [
    {
      "driver_id": "D123456",
      "speed_variance": 15.2,
      "harsh_braking": 3,
      "target": 0.672
    }
  ],
  "target_column": "target",
  "feature_columns": ["speed_variance", "harsh_braking"],
  "methods": ["shap", "permutation", "mutual_info"]
}
```

**Response:**
```json
{
  "analysis_id": "feature_importance_20241209_153000",
  "methods_used": ["shap", "permutation", "mutual_info"],
  "importance_scores": {
    "shap": {
      "speed_variance": 0.142,
      "harsh_braking": 0.089,
      "night_driving": 0.065
    },
    "permutation": {
      "speed_variance": 0.138,
      "harsh_braking": 0.092,
      "night_driving": 0.061
    },
    "mutual_info": {
      "speed_variance": 0.145,
      "harsh_braking": 0.087,
      "night_driving": 0.068
    }
  },
  "consensus_ranking": [
    "speed_variance",
    "harsh_braking", 
    "night_driving"
  ],
  "stability_score": 0.94
}
```

### POST `/analysis/model_comparison`
Compare multiple models on the same dataset.

**Request Body:**
```json
{
  "data": [
    {
      "duration": 365,
      "event": 1,
      "age": 25,
      "experience": 2.5
    }
  ],
  "models": ["cox", "weibull", "bayesian_hierarchical"],
  "comparison_metrics": ["c_index", "aic", "log_likelihood"]
}
```

**Response:**
```json
{
  "comparison_id": "model_comp_20241209_153000",
  "models_compared": ["cox", "weibull", "bayesian_hierarchical"],
  "performance": {
    "cox": {
      "c_index": 0.79,
      "aic": 2847.0,
      "log_likelihood": -1423.5
    },
    "weibull": {
      "c_index": 0.76,
      "aic": 2923.5,
      "log_likelihood": -1458.8
    },
    "bayesian_hierarchical": {
      "c_index": 0.82,
      "aic": null,
      "log_likelihood": null,
      "loo": -1425.3,
      "waic": -1423.8
    }
  },
  "ranking": {
    "by_c_index": ["bayesian_hierarchical", "cox", "weibull"],
    "by_aic": ["cox", "weibull"],
    "overall_best": "bayesian_hierarchical"
  }
}
```

---

## Error Responses

All endpoints return standardized error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data format",
    "details": "Field 'driver_id' is required",
    "timestamp": "2024-12-09T15:30:00Z",
    "request_id": "req_20241209_153000_abc123"
  }
}
```

### Error Codes
- `VALIDATION_ERROR` (400): Invalid request data
- `UNAUTHORIZED` (401): Invalid or missing API key
- `FORBIDDEN` (403): Insufficient permissions
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `MODEL_ERROR` (422): Model processing error
- `SERVER_ERROR` (500): Internal server error

---

## Rate Limiting Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1702135800
X-RateLimit-Window: 3600
```

---

## SDK Examples

### Python SDK
```python
from driver_analytics_client import DriverAnalyticsClient

client = DriverAnalyticsClient(api_key="your-api-key", base_url="http://localhost:8003")

# Calculate risk score
risk_result = client.scoring.calculate_risk({
    "driver_id": "D123456",
    "speed_variance": 15.2,
    "harsh_acceleration_events": 3
})

print(f"Risk Score: {risk_result.risk_score}")
```

### cURL Examples
```bash
# Calculate risk score
curl -X POST "http://localhost:8003/scoring/calculate_risk" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "driver_features": {
      "driver_id": "D123456",
      "speed_variance": 15.2,
      "harsh_acceleration_events": 3
    }
  }'

# Health check
curl -X GET "http://localhost:8003/health" \
  -H "X-API-Key: your-api-key"
```

---

## Webhooks

The API supports webhooks for real-time notifications:

### POST `/webhooks/register`
Register a webhook endpoint.

**Request Body:**
```json
{
  "url": "https://your-domain.com/webhook",
  "events": ["risk_alert", "model_update"],
  "secret": "webhook-secret"
}
```

### Webhook Events
- `risk_alert`: Triggered when driver risk exceeds threshold
- `model_update`: Triggered when models are retrained
- `batch_complete`: Triggered when batch processing completes

---

## Support

For API support and questions:
- **Email:** jguwalan@umd.edu
- **GitHub Issues:** [Driver-Behavior-Analytics-System/issues](https://github.com/JayDS22/Driver-Behavior-Analytics-System/issues)
- **Documentation:** [API Docs](https://jayds22.github.io/Portfolio/post/chapter-7/)

---

## Changelog

### v1.0.0 (2024-12-09)
- Initial API release
- Cox proportional hazards modeling
- Bayesian hierarchical models
- Real-time risk scoring
- SHAP explainability
- Comprehensive testing suite