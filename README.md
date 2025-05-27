# Text-to-Speech Application with Dia-1.6B

A full-stack text-to-speech application that uses the Dia-1.6B model for voice synthesis. Built with FastAPI backend and Svelte frontend.

## Features

- High-quality text-to-speech conversion using Dia-1.6B model
- Fast and responsive FastAPI backend
- Modern and intuitive Svelte frontend
- Real-time text-to-speech generation
- Deployed on Pruna and Koyeb for optimal performance

## Project Structure

```
text-to-voice/
├── backend/          # FastAPI server and Dia model implementation
│   ├── dia           # Model itself
│   ├── main.py       # ML model integration
│   └── pyproject.toml
├── frontend/         # Svelte frontend application
│   ├── src/          # Source code
│   ├── public/       # Static assets
│   └── package.json
└── README.md
└── tutorial.md

```

## Prerequisites

- Python 3.8+
- Node.js 16+
- pnpm
- uv (Python package installer)

## Setup Instructions

1. Clone the project:

   ```bash
   git clone <repository-url>

   ```

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend

   ```

2. Install dependencies using uv:

   ```bash
   uv sync

   ```

3. Start the FastAPI server:

   ```bash
   uv run fastapi dev main.py

   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend

   ```

2. Install dependencies:

   ```bash
   pnpm install

   ```

3. Start the development server:

   ```bash
   pnpm run dev

   ```

## Deployment

For deployment check the [tutorial](https://www.koyeb.com/tutorials/use-dia-1-6b-to-build-a-text-to-speech-application-on-serverless-gpus)

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure all dependencies are correctly installed
   - Check if the model files are in the correct location
2. **Frontend connection issues**
   - Verify the backend URL in the frontend configuration
   - Check if CORS is properly configured
3. **Audio generation problems**
   - Check the input text format
   - Verify the model parameters

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://www.notion.so/koyeb/LICENSE) file for details.

### Third-Party Components

This project incorporates the Dia TTS model:

- **Dia TTS Model**: Licensed under Apache License 2.0
  - Original repository: https://github.com/nari-labs/dia
  - Description: A TTS model capable of generating ultra-realistic dialogue in one pass
  - Copyright: nari-labs

The Dia model components in `backend/dia/` are based on or derived from the original Dia project. See the [NOTICE](https://www.notion.so/koyeb/NOTICE) file for detailed attribution information.
