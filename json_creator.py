import PyPDF2
import re
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LegalStatuteDatabase:
    """Database of legal statute patterns"""
    def __init__(self):
        self.acts_database = {
            "constitution": [
                r'CONSTITUTION OF INDIA',
                r'CONSTITUTIONAL',
                r'ARTICLE \d+(?:\([A-Za-z]\))?'
            ],
            "civil": [
                r'CIVIL PROCEDURE CODE',
                r'CPC',
                r'CODE OF CIVIL PROCEDURE'
            ],
            "criminal": [
                r'CRIMINAL PROCEDURE CODE',
                r'CRPC',
                r'CODE OF CRIMINAL PROCEDURE'
            ],
            "commercial": [
                r'COMPANIES ACT,?\s*\d{4}',
                r'PARTNERSHIP ACT',
                r'CONTRACT ACT'
            ],
            "arbitration": [
                r'ARBITRATION (?:AND CONCILIATION )?ACT,?\s*\d{4}',
                r'ARBITRATION ACT'
            ],
            "labor": [
                r'INDUSTRIAL DISPUTES ACT',
                r'FACTORIES ACT',
                r'LABOUR ACT'
            ],
            "ip": [
                r'PATENT[S]? ACT',
                r'COPYRIGHT ACT',
                r'TRADEMARK[S]? ACT'
            ],
            "tax": [
                r'INCOME TAX ACT',
                r'GST ACT',
                r'TAXATION ACT'
            ]
        }
        
        self.section_patterns = [
            r'SECTION \d+(?:\([A-Za-z]\))?',
            r'SEC\. \d+(?:\([A-Za-z]\))?',
            r'CLAUSE \d+(?:\([A-Za-z]\))?'
        ]

class LegalTextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text from PDF"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('‐', '-')
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Simpler sentence splitting that preserves legal references
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

class LegalStatuteExtractor:
    def __init__(self):
        self.preprocessor = LegalTextPreprocessor()
        self.statute_db = LegalStatuteDatabase()
        self.logger = logging.getLogger(__name__)
        self.unique_references = defaultdict(set)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        self.logger.warning(f"Page extraction error: {str(e)}")
                        continue
            
            return self.preprocessor.clean_text(text)
            
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            raise

    def extract_statutes_with_context(self, text: str) -> Dict:
        findings = defaultdict(list)
        sentences = self.preprocessor.split_into_sentences(text)
        
        for category, patterns in self.statute_db.acts_database.items():
            for pattern in patterns:
                for i, sentence in enumerate(sentences):
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        statute = match.group(0).upper()
                        reference_key = f"{category}:{statute.lower()}"
                        
                        if reference_key not in self.unique_references[category]:
                            self.unique_references[category].add(reference_key)
                            context = ' '.join(sentences[max(0, i-1):min(len(sentences), i+2)])
                            
                            findings[category].append({
                                'statute': statute,
                                'context': context,
                                'sentence_index': i
                            })
        
        # Extract sections
        seen_sections = set()
        sections = []
        for pattern in self.statute_db.section_patterns:
            for i, sentence in enumerate(sentences):
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    section = match.group(0).upper()
                    if section.lower() not in seen_sections:
                        seen_sections.add(section.lower())
                        context = ' '.join(sentences[max(0, i-1):min(len(sentences), i+2)])
                        sections.append({
                            'section': section,
                            'context': context,
                            'sentence_index': i
                        })
        
        findings['sections'] = sections
        return dict(findings)

    def process_legal_document(self, pdf_path: str) -> Dict:
        self.logger.info(f"Processing document: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        statutes = self.extract_statutes_with_context(text)
        
        return {
            "document_id": Path(pdf_path).stem,
            "file_name": Path(pdf_path).name,
            "processing_timestamp": datetime.now().isoformat(),
            "legal_references": {
                "constitutional": self._format_category(statutes.get("constitution", [])),
                "civil_laws": self._format_category(statutes.get("civil", [])),
                "criminal_laws": self._format_category(statutes.get("criminal", [])),
                "commercial_laws": self._format_category(statutes.get("commercial", [])),
                "arbitration": self._format_category(statutes.get("arbitration", [])),
                "labor_laws": self._format_category(statutes.get("labor", [])),
                "ip_laws": self._format_category(statutes.get("ip", [])),
                "tax_laws": self._format_category(statutes.get("tax", [])),
                "sections": self._format_category(statutes.get("sections", []))
            },
            "statistics": {
                "total_references": sum(len(refs) for refs in statutes.values()),
                "categories_found": [cat for cat, refs in statutes.items() if refs]
            }
        }

    @staticmethod
    def _format_category(entries: List[Dict]) -> List[Dict]:
        return sorted([{
            "reference": entry.get("statute", entry.get("section", "")),
            "context": entry["context"],
            "location": entry["sentence_index"]
        } for entry in entries], key=lambda x: x["location"])