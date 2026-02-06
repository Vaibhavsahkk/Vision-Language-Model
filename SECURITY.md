# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in this project, please report it privately via GitHub's Security Advisory feature or by emailing the maintainers directly.

**Please do not report security vulnerabilities through public GitHub issues.**

## Supported Versions

This is an educational/challenge project. Security updates will be provided on a best-effort basis.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Security Best Practices

### API Keys

- **Never commit API keys** to the repository
- Use the `.env` file for storing API keys locally
- Copy `.env.example` to `.env` and add your keys
- The `.env` file is in `.gitignore` to prevent accidental commits

### Before Running

1. Ensure your `.env` file contains your API keys
2. Never share your `.env` file publicly
3. Rotate API keys if accidentally exposed

### VLM API Usage

This project uses third-party VLM APIs (OpenAI, Google):
- Images sent to these APIs may be stored by the providers
- Review each provider's data usage policy before use
- Consider privacy implications when analyzing game images
- Do not use with sensitive or proprietary images

### Dependencies

- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Review dependency changes for security advisories
- Use virtual environments to isolate dependencies

## Known Limitations

- This is a demonstration/educational project
- Not designed for production use
- VLM APIs require internet connectivity
- API calls may incur costs based on provider pricing
