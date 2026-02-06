# Contributing to Cascadia Scorer

Thank you for your interest in contributing to this project!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cascadia-scorer.git
   cd cascadia-scorer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Add docstrings to all public functions/classes
- Keep functions focused and modular

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain or improve code coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit with clear, descriptive messages
6. Push to your fork
7. Open a Pull Request with a clear description

## Issues

- Check existing issues before creating a new one
- Provide clear reproduction steps for bugs
- Include system information (OS, Python version, etc.)

## Questions?

Feel free to open an issue for questions or discussions.
