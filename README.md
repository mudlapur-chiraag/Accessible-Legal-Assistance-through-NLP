eSRT - Legal Document Analyzer
A Python-based application for analyzing legal documents with advanced NLP capabilities and Kannada translation support.
Features

PDF document analysis
Legal entity extraction
Statute identification
Case summarization
Similar case finding
Kannada translation support
Interactive GUI interface

Setup

Install dependencies:

bashCopypip install -r requirements.txt

Configure API keys:


Add Sarvam.ai API key for translation services


Install model files:


Download required models to models/ directory:

en_legal_ner_trf
legal-pegasus-finetuned-final



Project Structure
Copy.
├── __init__.py
├── app.ipynb           # Jupyter notebook for development
├── database_manager.py # Database operations
├── gui.py             # PyQt5 GUI implementation
├── json_creator.py    # JSON processing utilities
├── legal_processor.py # Core document processing
├── main.ipynb         # Application entry point
└── datasets/          # Document storage
    ├── documents/     # PDF storage
    └── json_data/     # Processed data
Usage
Run the application:
pythonCopypython main.py
License
MIT
Contributors

Chiraag Mudlapur            PES2UG21CS146
Diya Mariya                 PES2UG21CS167
Prashil Himanshu Jatakiya   PES2UG21CS211
Abhilash Vinod              PES2UG21CS014
