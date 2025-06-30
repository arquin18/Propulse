# Propulse Backend

The backend service is built with FastAPI and handles all the core functionality of the Propulse system, including agent coordination, vector database management, and API endpoints.

## 📁 Directory Structure

```
backend/
├── agents/           # Agent implementations
│   ├── retriever/    # Retriever agent logic
│   ├── writer/       # Writer agent logic
│   └── verifier/     # Verifier agent logic
├── api/             # API endpoints and routes
│   ├── v1/          # API version 1
│   └── middleware/  # Custom middleware
├── core/            # Core business logic
│   ├── config/      # Configuration
│   ├── models/      # Data models
│   └── services/    # Business services
├── logs/            # Log files
└── main.py          # Application entry point
```

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   conda activate propulse
   ```

2. **Start the Server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

3. **View API Documentation**
   - OpenAPI UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 🔧 Configuration

The backend service is configured through environment variables:

- `GEMINI_API_KEY`: Google Gemini API key
- `VECTOR_DB_RFP_PATH`: Path to RFP vector database
- `VECTOR_DB_PROPOSAL_PATH`: Path to proposal vector database
- `LOG_LEVEL`: Logging level (default: INFO)

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app tests/
```

## 📝 API Documentation

### Key Endpoints

- `POST /api/v1/proposals/generate`
  - Generate proposal from prompt
  - Accepts prompt text and optional RFP document

- `GET /api/v1/proposals/{proposal_id}`
  - Get proposal generation status
  - Returns current status and progress

- `GET /api/v1/proposals/{proposal_id}/download`
  - Download generated proposal
  - Returns document in requested format

## 🔍 Logging

Logs are stored in the `logs/` directory:
- `app.log`: Application logs
- `error.log`: Error logs
- `access.log`: API access logs

## 🛠️ Development

1. **Code Style**
   ```bash
   black .
   isort .
   flake8
   ```

2. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Adding New Endpoints**
   - Create new route in `api/v1/`
   - Update OpenAPI documentation
   - Add tests in `tests/`

## 🔒 Security

- JWT authentication
- Rate limiting
- Input validation
- CORS configuration
- Security headers

## 🐛 Debugging

1. Enable debug mode:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. Check logs:
   ```bash
   tail -f logs/app.log
   ```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Google Gemini API Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) 