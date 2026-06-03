# CSE440 NLP Project: AI-Powered Text Summarization

## Overview

This project implements an intelligent text summarization solution leveraging Claude AI to automatically generate clear, concise executive summaries from input documents. Built for efficiency and accuracy, this tool is designed to streamline information processing workflows in corporate environments.

## Key Features

- **Dynamic File Processing**: Reads and processes text files seamlessly
- **AI-Powered Summarization**: Uses Claude AI for high-quality, context-aware summaries
- **Executive Summary Generation**: Produces clear, actionable summaries suitable for business presentations
- **Scalable Architecture**: Designed to handle multiple documents and batch processing

## Tech Stack

- **Language**: Python
- **AI Model**: Claude AI (via Anthropic API)
- **Use Case**: Natural Language Processing, Text Analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Anthropic API key
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qewCello/CSE440-NLP-project.git
   cd CSE440-NLP-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

### Usage

Run the summarization script:
```bash
python summarize.py --input <input_file.txt> --output <summary_output.txt>
```

## Project Structure

```
CSE440-NLP-project/
├── README.md
├── requirements.txt
├── summarize.py          # Main summarization script
└── sample_inputs/        # Example input documents
```

## Use Cases

- **Business Intelligence**: Quickly extract insights from lengthy reports
- **Research**: Summarize academic papers and research documents
- **Knowledge Management**: Process and organize information at scale
- **Content Curation**: Prepare summaries for stakeholder communications

## Performance Metrics

- Fast processing time for documents up to 10,000 tokens
- High accuracy in capturing key information and context
- Optimized for business-relevant summaries

## Future Enhancements

- [ ] Support for multiple input formats (PDF, DOCX, etc.)
- [ ] Customizable summary length and detail levels
- [ ] Batch processing capabilities
- [ ] Web interface for easier access
- [ ] Integration with document management systems

## Contributing

Contributions are welcome! Please follow standard Git workflow practices and ensure code quality before submitting pull requests.

## License

This project is part of CSE 440 coursework.

## Contact & Support

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

**Last Updated**: 2026-06-03
