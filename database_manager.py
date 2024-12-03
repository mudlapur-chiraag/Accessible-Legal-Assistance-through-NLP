# database_manager.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.documents_path = self.base_path / "documents"
        self.json_path = self.base_path / "json_data"
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self.json_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.case_database: Dict = {}
        self.load_case_database()

    def load_case_database(self):
        """Load all JSON case data into memory"""
        try:
            for json_file in self.json_path.glob("*.json"):
                if json_file.name != "database_index.json":  # Skip index file
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                            
                            # Store the base PDF path (without _analysis)
                            pdf_filename = json_file.stem.replace('_analysis', '') + '.pdf'
                            pdf_path = self.documents_path / pdf_filename
                            if pdf_path.exists():
                                case_data['pdf_path'] = str(pdf_path)
                                
                            self.case_database[json_file.stem] = case_data
                    except Exception as e:
                        self.logger.error(f"Error loading {json_file}: {str(e)}")
                
            self.logger.info(f"Loaded {len(self.case_database)} cases into database")
        except Exception as e:
            self.logger.error(f"Error loading case database: {str(e)}")

    def save_case_data(self, case_id: str, data: Dict) -> bool:
        """Save case data to JSON file"""
        try:
            json_path = self.json_path / f"{case_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update in-memory database
            self.case_database[case_id] = data
            return True
        except Exception as e:
            self.logger.error(f"Error saving case data: {str(e)}")
            return False

    def get_case_data(self, case_id: str) -> Optional[Dict]:
        """Retrieve case data from database"""
        return self.case_database.get(case_id)

    def save_document(self, document_path: str, case_id: str) -> bool:
        """Save document file to documents directory"""
        try:
            src_path = Path(document_path)
            dst_path = self.documents_path / f"{case_id}{src_path.suffix}"
            
            # Copy document to documents directory
            with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                dst.write(src.read())
            return True
        except Exception as e:
            self.logger.error(f"Error saving document: {str(e)}")
            return False

    def get_similar_cases(self, query_data: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar cases without storing the input case"""
        try:
            results = []
            query_references = self._extract_references(query_data)
            
            # Only search through existing database, don't add new case
            for case_id, case_data in self.case_database.items():
                case_references = self._extract_references(case_data)
                similarity_score = self._calculate_similarity(query_references, case_references)
                
                if similarity_score > 0:
                    results.append({
                        'case_id': case_id,
                        'similarity_score': similarity_score,
                        'total_matches': len(self._get_matching_references(
                            query_references, case_references
                        )),
                        'matching_references': self._get_matching_references(
                            query_references, case_references
                        )
                    })
            
            results.sort(
                key=lambda x: (x['similarity_score'], x['total_matches']), 
                reverse=True
            )
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar cases: {str(e)}")
            return []

    def _extract_references(self, data: Dict) -> Dict[str, set]:
        """Extract all legal references from case data"""
        references = {}
        
        # Extract from legal_references if it exists
        if 'legal_references' in data:
            for category, refs in data['legal_references'].items():
                if isinstance(refs, list):
                    references[category] = {
                        ref['reference'].upper() 
                        for ref in refs 
                        if 'reference' in ref
                    }
        
        return references

    def _calculate_similarity(self, query_refs: Dict, case_refs: Dict) -> float:
        """Calculate similarity score between two sets of legal references"""
        if not query_refs or not case_refs:
            return 0.0
            
        total_matches = 0
        total_refs = 0
        
        for category in query_refs:
            if category in case_refs:
                matches = len(query_refs[category] & case_refs[category])
                total_matches += matches
            total_refs += len(query_refs[category])
            
        return round(total_matches / total_refs, 3) if total_refs > 0 else 0.0

    def _get_matching_references(self, query_refs: Dict, case_refs: Dict) -> Dict[str, List[str]]:
        """Get all matching references by category"""
        matches = {}
        
        for category in query_refs:
            if category in case_refs:
                common_refs = query_refs[category] & case_refs[category]
                if common_refs:
                    matches[category] = sorted(common_refs)
                    
        return matches
    
    def get_pdf_path(self, case_id: str) -> Optional[str]:
        """Get the PDF file path for a given case ID"""
        try:
            # Remove '_analysis' suffix if present for finding the PDF
            pdf_filename = case_id.replace('_analysis', '') + '.pdf'
            pdf_path = self.documents_path / pdf_filename
            
            if pdf_path.exists():
                return str(pdf_path)
                
            # Try without .pdf extension in case the file has a different extension
            base_path = self.documents_path / case_id.replace('_analysis', '')
            for ext in ['.PDF', '.pdf']:
                if (base_path.parent / (base_path.name + ext)).exists():
                    return str(base_path.parent / (base_path.name + ext))
                    
            return None
        except Exception as e:
            self.logger.error(f"Error getting PDF path: {str(e)}")
            return None
    
    def save_case_data(self, case_id: str, data: Dict, pdf_path: Optional[str] = None) -> bool:
        """Only save if case is from dataset, not input"""
        # Skip saving if case_id starts with CASE_ (input document)
        if case_id.startswith('CASE_'):
            return True
            
        try:
            if pdf_path:
                data['pdf_path'] = str(pdf_path)
                
            json_path = self.json_path / f"{case_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update in-memory database
            self.case_database[case_id] = data
            return True
        except Exception as e:
            self.logger.error(f"Error saving case data: {str(e)}")
            return False