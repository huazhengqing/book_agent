# Technology Stack

## Core Framework
- **Python 3.6+**: Primary programming language
- **LiteLLM**: Multi-provider LLM integration with caching support
- **Mem0**: Memory management and persistent context storage
- **LangChain**: Text processing and document handling

## Infrastructure Components
- **Docker Compose**: Container orchestration for services
- **Memgraph**: Graph database for knowledge relationships
- **Qdrant**: Vector database for semantic search
- **ChromaDB**: Alternative vector storage option

## Key Dependencies
- **OpenAI/Anthropic/Google**: LLM providers via LiteLLM
- **Neo4j**: Graph database connectivity
- **Trafilatura**: Web content extraction
- **DuckDuckGo Search**: Web search integration
- **Gradio**: Development UI interface

## Build & Development Commands

### Environment Setup
```bash
# Install dependencies and setup virtual environment
./install_env.sh

# Setup Docker services
./install_docker.sh
./install_memgraph.sh
./install_qdrant.sh

# Start services
docker-compose up -d
```

### Running the System
```bash
# Activate environment
source venv/bin/activate

# Story writing mode
python3 recursive/engine.py --filename test_data/story.jsonl --output-filename output.jsonl --model openai/deepseek-ai/DeepSeek-R1-0528 --mode story --language zh

# Book writing mode  
python3 recursive/engine.py --mode book --language en

# Report writing mode
python3 recursive/engine.py --mode report --language en
```

### Configuration
- API keys: `recursive/api_key.env`
- Docker services: `docker-compose.yml`
- Dependencies: `requirements.txt`
- Package setup: `setup.py`

## Development Notes
- Uses setuptools for package management
- Supports both development (`pip install -e .`) and production installs
- Requires Docker for Memgraph and Qdrant services
- Environment variables loaded from `.env` files