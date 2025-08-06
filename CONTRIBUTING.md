# Contributing to AI Evals Tutorial

Thank you for your interest in contributing to the AI Evals Tutorial! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check existing issues to avoid duplicates
2. Use the issue templates when available
3. Include relevant details:
   - Module/section affected
   - Expected vs actual behavior
   - Steps to reproduce
   - Environment details (Python version, OS, etc.)

### Submitting Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/ai-evals-tutorial.git
   cd ai-evals-tutorial
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation
   - Test your changes thoroughly

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Add meaningful commit message"
   ```
   
   Use conventional commits:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Formatting changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

## Contribution Areas

### Content Contributions
- **Exercises**: Add new hands-on exercises
- **Case Studies**: Share real-world implementation experiences
- **Templates**: Create reusable evaluation frameworks
- **Documentation**: Improve explanations and examples

### Code Contributions
- **Bug Fixes**: Help fix identified issues
- **Feature Additions**: Add new evaluation tools or methods
- **Performance Improvements**: Optimize existing code
- **Test Coverage**: Add or improve tests

### Community Contributions
- **Translations**: Help translate content to other languages
- **Reviews**: Review and provide feedback on PRs
- **Support**: Help answer questions in discussions
- **Promotion**: Share the tutorial with others who might benefit

## Style Guidelines

### Python Code Style
- Follow PEP 8
- Use type hints where appropriate
- Include docstrings for functions and classes
- Maximum line length: 88 characters (Black formatter)

### Markdown Style
- Use clear, concise language
- Include code examples where relevant
- Add diagrams for complex concepts
- Structure content with appropriate headers

### Commit Messages
- Keep the subject line under 50 characters
- Use imperative mood ("Add feature" not "Added feature")
- Reference issues when applicable (#123)

## Testing

Before submitting:
1. Run existing tests: `pytest`
2. Check code formatting: `black . --check`
3. Lint your code: `pylint modules/`
4. Verify documentation builds: `mkdocs build`

## Getting Help

- **Discord**: Join our community (link coming soon)
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Open an issue for bugs or feature requests

## Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Given credit in relevant sections

Thank you for helping make AI evaluation accessible to everyone!