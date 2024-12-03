import spacy
import re
import warnings
import PyPDF2
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import requests
from difflib import SequenceMatcher
import numpy as np
import language_tool_python
from pathlib import Path
from datetime import datetime
from database_manager import DatabaseManager
import logging

class SemanticChunker:
    def __init__(self, tokenizer, max_tokens: int = 1000, overlap_tokens: int = 100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer
        
    def split_into_sentences(self, text: str) -> List[str]:
        legal_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence endings
            r'(?<=\d\.)\s+(?=[A-Z])',    # Numbered points
            r'(?<=\]\.)\s+(?=[A-Z])',    # Citations
            r'(?<=\}\.)\s+(?=[A-Z])'     # Legal references
        ]
        pattern = '|'.join(legal_patterns)
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
        
    def create_chunks(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_buffer = []
        
        for para in paragraphs:
            sentences = self.split_into_sentences(para)
            
            for sent in sentences:
                sent_tokens = len(self.tokenizer.encode(sent))
                
                if current_tokens + sent_tokens > self.max_tokens:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(chunk_text)
                        
                        overlap_start = max(0, len(current_chunk) - 2)
                        overlap_buffer = current_chunk[overlap_start:]
                        
                        current_chunk = overlap_buffer + [sent]
                        current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
class RecursiveSummarizer:
    def __init__(self, model, tokenizer, chunker):
        self.model = model
        self.tokenizer = tokenizer
        self.chunker = chunker

    def post_process_summary(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?i)\b(the )\1+', r'\1', text)
        text = re.sub(r'(?i)\b(that )\1+', r'\1', text)
        
        sentences = text.split('. ')
        processed_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            normalized = ' '.join(sorted(sentence.lower().split()))
            if normalized not in seen_content:
                seen_content.add(normalized)
                processed_sentences.append(sentence)
        
        text = '. '.join(processed_sentences)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\s+([.,])', r'\1', text)
        
        return text
        
    def summarize_text(self, text: str, target_length: int) -> str:
        current_tokens = len(self.tokenizer.encode(text))
        
        if current_tokens <= target_length:
            return text
            
        chunks = self.chunker.create_chunks(text)
        chunk_summaries = []
        
        for chunk in chunks:
            inputs = self.tokenizer(chunk, max_length=1024, truncation=True, 
                                  padding=True, return_tensors="pt").to(self.model.device)
            
            chunk_token_count = len(inputs['input_ids'][0])
            chunk_summary_length = max(50, min(chunk_token_count // 4, target_length // len(chunks)))
            min_length = min(30, chunk_summary_length - 10)
            
            summary_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=4,
                min_length=30,
                max_length=chunk_summary_length,
                length_penalty=2.0,
                early_stopping=True,
                temperature=0.7,  # Add temperature for more diversity
                top_p=0.9,       # Add nucleus sampling
                do_sample=True   # Enable sampling instead of pure beam search
            )
            
            chunk_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(chunk_summary)
        
        combined_summary = remove_redundant_text(chunk_summaries)
        combined_summary = self.post_process_summary(combined_summary)
        current_tokens = len(self.tokenizer.encode(combined_summary))
        
        if current_tokens > target_length:
            return self.summarize_text(combined_summary, target_length)
            
        return combined_summary

class OllamaProcessor:
    def __init__(self, model_name: str = "tejasmankweshwar/legalsathi"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.system_prompt = """You are Legal-Sathi, your trusted Legal-AI-Buddy specializing in Indian law. Visualize me as your virtual guide through the legal landscape of India. I am here to assist you in understanding and navigating the intricacies of Indian legal matters with ease."""
        print(f"Initialized OllamaProcessor with model: {model_name}")
        
    def generate(self, prompt: str) -> str:
        try:
            print("\n=== Sending request to Legal-Sathi ===")
            print(f"Prompt: {prompt[:1000]}...")
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "temperature": 0.2,
                    "top_p": 0.9
                }
            )
            response.raise_for_status()
            result = response.json()['response']
            
            print("\n=== Received response from Legal-Sathi ===")
            print(f"Response: {result[:200]}...")
            return result
        except Exception as e:
            print(f"\n=== Error with Legal-Sathi: {e} ===")
            return None

class LegalEntityProcessor:
    def __init__(self):
        self.common_prefixes = ['mr', 'mrs', 'ms', 'dr', 'hon', 'honble', 'shri', 'smt']
        self.excluded_words = {
            'also', 'the', 'and', 'or', 'but', 'if', 'then', 'thus', 
            'therefore', 'hence', 'whereas', 'pursuant', 'accordingly'
        }
        self.designation_patterns = [
            r'\b(?:justice|judge|advocate|adv|solicitor)\b',
            r'\b(?:chief|senior|junior|principal)\s+(?:justice|judge|advocate)\b',
            r'\b(?:petitioner|respondent|appellant|plaintiff|defendant)\b'
        ]
        self.statute_patterns = [
            r'\b\d{4}\s+Act\b',
            r'\b(?:Section|Sec\.|§)\s*\d+(?:\([a-zA-Z0-9]+\))?\s+of\s+the\s+[A-Za-z\s]+Act',
            r'\b[A-Z][A-Za-z\s]+Act,?\s+\d{4}\b',
            r'\b(?:Article|Art\.)\s*\d+(?:\([a-zA-Z0-9]+\))?\s+of\s+the\s+Constitution'
        ]
        
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        for prefix in self.common_prefixes:
            text = re.sub(rf'\b{prefix}\.\s*', '', text)
            text = re.sub(rf'\b{prefix}\s+', '', text)
        for pattern in self.designation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def are_similar_entities(self, entity1: str, entity2: str, threshold: float = 0.85) -> bool:
        norm1 = self.normalize_text(entity1)
        norm2 = self.normalize_text(entity2)
        
        if norm1 == norm2:
            return True
        
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold

    def process_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        processed_entities = defaultdict(list)
        
        for entity_type, entity_list in entities.items():
            entity_freq = defaultdict(int)
            for entity in entity_list:
                entity_freq[entity.strip()] += 1
                
            sorted_entities = sorted(
                set(entity_list),
                key=lambda x: (entity_freq[x.strip()], len(x)),
                reverse=True
            )
            
            final_entities = []
            for entity in sorted_entities:
                # Skip if entity is in excluded words, too short, or just numbers
                if (entity.strip().lower() in self.excluded_words or 
                    len(entity.strip()) < 3 or 
                    entity.strip().isdigit()):
                    continue
                    
                is_duplicate = any(
                    self.are_similar_entities(entity, existing_entity)
                    for existing_entity in final_entities
                )
                
                if not is_duplicate:
                    final_entities.append(entity.strip())
                    
            processed_entities[entity_type] = final_entities
            
        return dict(processed_entities)

    def extract_statutes(self, text: str) -> List[str]:
        statutes = []
        for pattern in self.statute_patterns:
            matches = re.finditer(pattern, text, flags=re.IGNORECASE)
            statutes.extend(match.group().strip() for match in matches)
        
        seen = set()
        return [x for x in statutes if not (x.lower() in seen or seen.add(x.lower()))]
    
def remove_redundant_text(summaries: List[str], similarity_threshold: float = 0.8) -> str:
    def get_sentences(text):
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def is_similar(sent1: str, sent2: str) -> bool:
        return SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio() > similarity_threshold
    
    all_sentences = []
    for summary in summaries:
        sentences = get_sentences(summary)
        for sent in sentences:
            is_redundant = any(is_similar(sent, existing_sent) 
                             for existing_sent in all_sentences)
            if not is_redundant:
                all_sentences.append(sent)
    
    return '. '.join(all_sentences) + '.'

class LegalDocumentProcessor:
    def __init__(self, ner_model_path: str, summarizer_model_path: str):
        self.nlp = spacy.load(ner_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PegasusTokenizer.from_pretrained(summarizer_model_path)
        self.model = PegasusForConditionalGeneration.from_pretrained(summarizer_model_path).to(self.device)
        self.entity_processor = LegalEntityProcessor()
        self.llm = OllamaProcessor()
        self.chunker = SemanticChunker(tokenizer=self.tokenizer)
        self.summarizer = RecursiveSummarizer(self.model, self.tokenizer, self.chunker)
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.unique_references = defaultdict(set)
        self.db_manager = DatabaseManager()
        warnings.filterwarnings('ignore')

    def correct_grammar(self, text: str) -> str:
        matches = self.grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with improved formatting preservation."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # Add consistent page numbering and formatting
                    page_text = f"- {page_num + 1} -\n" + page_text
                    
                    # Clean up text
                    page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)  # Add space between joined words
                    page_text = re.sub(r'\s*\n\s*', '\n', page_text)  # Normalize line breaks
                    page_text = re.sub(r'\s{2,}', ' ', page_text)  # Remove multiple spaces
                    
                    text += page_text + "\n"
                    
                    # Add separation between pages
                    if page_num < total_pages - 1:
                        text += "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        chunks = self.chunker.create_chunks(text)  # Use semantic chunker instead
        all_entities = defaultdict(list)
        
        for chunk in chunks:
            doc = self.nlp(chunk)
            for ent in doc.ents:
                if ent.label_ not in all_entities:
                    all_entities[ent.label_] = []
                if ent.text.strip() not in all_entities[ent.label_]:
                    all_entities[ent.label_].append(ent.text.strip())
        
        return dict(all_entities)

    def extract_case_sections(self, text: str) -> Tuple[str, str]:
        """Extract judgment and order sections while preserving dates"""
        # Date pattern for preservation
        date_pattern = r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b'
        
        def preserve_dates(text: str) -> Tuple[str, Dict[str, str]]:
            dates = re.finditer(date_pattern, text, re.IGNORECASE)
            preserved = {}
            processed_text = text
            for i, match in enumerate(dates):
                key = f"__DATE_{i}__"
                preserved[key] = match.group()
                processed_text = processed_text.replace(match.group(), key)
            return processed_text, preserved

        def restore_dates(text: str, preserved: Dict[str, str]) -> str:
            restored_text = text
            for key, date in preserved.items():
                restored_text = restored_text.replace(key, date)
            return restored_text

        # Preserve dates before processing
        processed_text, preserved_dates = preserve_dates(text)
        
        # [Previous patterns and extraction logic]
        judgment_patterns = [
            r'JUDGMENT\s*:?[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))',
            r'JUDGEMENT\s*:?[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))',
            r'OPINION\s*:?[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))',
            r'This application under Section[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))',
            r'This petition[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))',
            r'This appeal[\s\S]+?(?=\n\s*(?:ORDER|Sd/-|TYPED BY|$))'
        ]
        
        order_patterns = [
            r'O\s*R\s*D\s*E\s*R\s*:?[\s\S]+?(?=\n\s*(?:Sd/-|TYPED BY|$))',
            r'FINAL ORDER\s*:?[\s\S]+?(?=\n\s*(?:Sd/-|TYPED BY|$))',
            r'OPERATIVE PORTION\s*:?[\s\S]+?(?=\n\s*(?:Sd/-|TYPED BY|$))',
            r'In view of the above[\s\S]+?(?=\n\s*(?:Sd/-|TYPED BY|$))',
            r'The following order[\s\S]+?(?=\n\s*(?:Sd/-|TYPED BY|$))'
        ]
        
        judgment_section = ""
        order_section = ""
        
        # Extract sections
        for pattern in judgment_patterns:
            match = re.search(pattern, processed_text, re.IGNORECASE)
            if match:
                judgment_section = match.group()
                break
                
        for pattern in order_patterns:
            match = re.search(pattern, processed_text, re.IGNORECASE)
            if match:
                order_section = match.group()
                break

        # Clean and restore dates
        if judgment_section:
            judgment_section = re.sub(r'\s+', ' ', judgment_section).strip()
            judgment_section = restore_dates(judgment_section, preserved_dates)
            judgment_section = f"<judgment>\n{judgment_section}\n</judgment>"
            
        if order_section:
            order_section = re.sub(r'\s+', ' ', order_section).strip()
            order_section = restore_dates(order_section, preserved_dates)
            order_section = f"<order>\n{order_section}\n</order>"

        return judgment_section, order_section

    def process_document(self, pdf_path: str) -> Dict:
        try:
            # Get raw text first
            raw_text = self.extract_text_from_pdf(pdf_path)
            entities = self.extract_named_entities(raw_text)
            cleaned_entities = self.entity_processor.process_entities(entities)
            
            statutes = self.entity_processor.extract_statutes(raw_text)
            cleaned_entities['STATUTES'] = statutes
            
            # Get sections for summarizer
            judgment_section, order_section = self.extract_case_sections(raw_text)
            
            # Generate initial summaries with Pegasus
            judgment_summary = self.generate_summary(judgment_section) if judgment_section else None
            order_summary = self.generate_summary(order_section) if order_section else None
            
            # Refine with Legal-Sathi
            if judgment_summary:
                judgment_summary = self.refine_summary_with_llama(raw_text, judgment_summary)
                
            if order_summary:
                order_summary = self.refine_summary_with_llama(raw_text, order_summary)
            
            temp_case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
            case_data = {
                "document_id": temp_case_id,
                "file_name": Path(pdf_path).name,
                "processing_date": datetime.now().isoformat(),
                "entities": cleaned_entities,
                "legal_references": {
                    "statutes": [{"reference": s} for s in statutes],
                    "sections": self._extract_section_references(raw_text)
                },
                "sections": {
                    "judgment_summary": judgment_summary,
                    "order_summary": order_summary
                }
            }
            
            # Get similar cases without storing input
            similar_cases = self.db_manager.get_similar_cases(case_data)
            case_data['similar_cases'] = similar_cases
            
            # Don't save the input case data
            return case_data
                    
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise

    def _extract_section_references(self, text: str) -> List[Dict]:
        """Extract section references from text"""
        sections = []
        section_pattern = r'(?:Section|Sec\.|§)\s*(\d+(?:\([a-zA-Z0-9]+\))?)'
        
        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        seen_sections = set()
        
        for match in matches:
            section = match.group(0).upper()
            if section not in seen_sections:
                seen_sections.add(section)
                sections.append({"reference": section})
                
        return sections

    def clean_text(self, text: str) -> str:
        """Clean and normalize legal text with improved preservation of structure."""
        # Fix ordinal numbers
        text = re.sub(r'(\d+)\s*(st|nd|rd|th)', r'\1\2', text)
        
        # Fix spacing around special characters
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s*([^\s])', r'\1 \2', text)
        
        # Normalize case numbers
        text = re.sub(r'(?i)no\.\s*', 'No.', text)
        
        # Clean up whitespace while preserving structure
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve special legal characters
        text = re.sub(r'[^\w\s.,;:()\[\]\/\-\'\"§¶]', ' ', text)
        
        # Clean up numbered points while preserving structure
        text = re.sub(r'(?<!\d)\d+\.\s+', '', text)  # Remove standalone numbers
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def generate_summary(self, text: str, max_length: int = 250) -> str:
        input_tokens = len(self.tokenizer.encode(text))
        target_length = max(50, input_tokens // 4)
        return self.summarizer.summarize_text(text, target_length)
    
    def post_process_summary(self, text: str) -> str:
        # Fix common issues
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'(?i)\b(the )\1+', r'\1', text)  # Remove repeated "the"
        text = re.sub(r'(?i)\b(that )\1+', r'\1', text) # Remove repeated "that"
        
        # Fix sentence structure
        sentences = text.split('. ')
        processed_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            # Normalize sentence for comparison
            normalized = ' '.join(sorted(sentence.lower().split()))
            if normalized not in seen_content:
                seen_content.add(normalized)
                processed_sentences.append(sentence)
        
        text = '. '.join(processed_sentences)
        
        # Clean up punctuation
        text = re.sub(r'\.+', '.', text)  # Remove multiple periods
        text = re.sub(r'\s+([.,])', r'\1', text)  # Fix spacing around punctuation
        
        return text

    def refine_summary_with_llama(self, raw_text: str, summary: str) -> str:
        try:
            prompt = f"""As Legal-Sathi, analyze this court document and enhance the initial summary.

    Court Document Text:
    {raw_text}...

    Initial Summary:
    {summary}

    Please provide an improved summary that:
    1. Identifies key legal provisions and citations from Indian law
    2. Explains the main legal arguments presented
    3. Clarifies the court's reasoning
    4. Highlights the final orders and their implications
    5. Retains accuracy without introducing new information

    Remember to maintain focus on Indian legal context. Provide your enhanced summary:"""

            refined_summary = self.llm.generate(prompt)
            return refined_summary if refined_summary else summary
        except Exception as e:
            logging.error(f"Error refining summary with Legal-Sathi: {e}")
            return summary
        
    def extract_statutes_with_context(self, text: str) -> Dict:
        findings = defaultdict(list)
        sentences = self.clean_text(text).split('. ')
        
        # Simplified statute patterns for demonstration
        patterns = {
            "constitution": [r'CONSTITUTION OF INDIA', r'ARTICLE \d+'],
            "civil": [r'CIVIL PROCEDURE CODE', r'CPC'],
            "criminal": [r'CRIMINAL PROCEDURE CODE', r'CRPC'],
            "sections": [r'SECTION \d+', r'SEC\. \d+']
        }
        
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                for i, sentence in enumerate(sentences):
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        statute = match.group(0).upper()
                        findings[category].append({
                            'statute': statute,
                            'context': sentence,
                            'sentence_index': i
                        })
        
        return dict(findings)

    def find_similar_cases(self, current_text: str, case_database: Dict) -> List[Dict]:
        current_statutes = self.extract_statutes_with_context(current_text)
        results = []
        
        for case_id, case_data in case_database.items():
            matches = self._compare_statutes(current_statutes, case_data)
            if matches['total_matches'] > 0:
                results.append({
                    'case_id': case_id,
                    'similarity_score': matches['similarity_score'],
                    'total_matches': matches['total_matches'],
                    'matching_references': matches['matching_references']
                })
                
        results.sort(key=lambda x: (x['similarity_score'], x['total_matches']), reverse=True)
        return results[:5]  # Return top 5 similar cases

    def _compare_statutes(self, current_statutes: Dict, case_data: Dict) -> Dict:
        total_matches = 0
        matching_references = defaultdict(list)
        
        for category in current_statutes:
            if category in case_data:
                current_refs = {ref['statute'] for ref in current_statutes[category]}
                case_refs = {ref['statute'] for ref in case_data[category]}
                matches = current_refs & case_refs
                if matches:
                    total_matches += len(matches)
                    matching_references[category].extend(sorted(matches))
        
        query_total = sum(len(refs) for refs in current_statutes.values())
        similarity_score = total_matches / query_total if query_total > 0 else 0
        
        return {
            'total_matches': total_matches,
            'similarity_score': round(similarity_score, 3),
            'matching_references': dict(matching_references)
        }