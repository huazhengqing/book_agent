# Project Structure

## Root Directory
- `docker-compose.yml`: Service orchestration (Memgraph, Qdrant)
- `requirements.txt`: Python dependencies
- `setup.py`: Package configuration
- `install_*.sh`: Setup scripts for environment and services

## Core Module (`recursive/`)
The main application logic organized by functionality:

### Core Engine
- `engine.py`: Main execution engine with `GraphRunEngine` class
- `graph.py`: Task graph management, node definitions, and status handling
- `memory.py`: Memory management and information collection
- `cache.py`: Caching layer for performance optimization

### Agent System (`recursive/agent/`)
- Agent proxy and orchestration
- Task-specific agents (planning, execution, reflection)
- Prompt templates organized by writing mode and language

### LLM Integration (`recursive/llm/`)
- LiteLLM proxy for multi-provider support
- Model configuration and API management

### Execution Layer (`recursive/executor/`)
- Task execution implementations
- Action handlers for different task types

### Memory & Storage
- `mem0_wrapper.py`: Mem0 integration for persistent memory
- Vector storage and retrieval logic

### Utilities (`recursive/utils/`)
- Display functions for graph visualization
- Helper functions and common utilities

## Configuration Files
- `recursive/api_key.env`: API credentials (not in repo)
- `recursive/api_key.env.example`: Template for API configuration

## Data & Testing
- `test_data/`: Sample input files (JSONL format)
- `docs/`: Documentation and logs
- `bak/`: Backup directory

## Architecture Patterns

### Task Hierarchy
- Root tasks decompose into subtasks recursively
- Each task has dependencies, status, and execution context
- Graph-based execution with topological sorting

### Agent Workflow
1. **Planning**: Decompose tasks into executable subtasks
2. **Execution**: Execute individual tasks with context
3. **Reflection**: Validate and refine results
4. **Aggregation**: Combine results into final output

### Memory Management
- Persistent storage via Mem0 and vector databases
- Context-aware retrieval for task execution
- Hierarchical information organization

### File Naming Conventions
- Snake_case for Python modules
- Descriptive names indicating functionality
- Language-specific suffixes for prompts (e.g., `_zh`, `_en`)