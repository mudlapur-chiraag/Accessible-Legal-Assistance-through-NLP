import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QTextEdit, 
                           QTabWidget, QTreeWidget, QTreeWidgetItem, QProgressBar,
                           QMessageBox, QHBoxLayout, QSplitter, QStackedWidget,
                           QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import requests
import subprocess
import platform
from pathlib import Path

class TranslationThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, text_to_translate):
        super().__init__()
        self.text_to_translate = text_to_translate

    def run(self):
        try:
            url = "https://api.sarvam.ai/translate"
            payload = {
                "input": self.text_to_translate,
                "source_language_code": "en-IN",
                "target_language_code": "kn-IN",
                "enable_preprocessing": True
            }
            headers = {
                "api-subscription-key": "e3594ef0-d4b4-44e4-86ee-8f9fe5a2bb13",
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            translated_text = response.json()["translated_text"]
            self.finished.emit(translated_text)
        except Exception as e:
            self.error.emit(str(e))

class ProcessingThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, processor, file_path):
        super().__init__()
        self.processor = processor
        self.file_path = file_path

    def run(self):
        try:
            self.progress.emit("Processing document...")
            results = self.processor.process_document(self.file_path)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class LegalAnalyzerGUI(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.current_summary = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('eSRT - Legal Document Analyzer')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
            }
            QPushButton#fileButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton#fileButton:hover {
                background-color: #2563eb;
            }
            QPushButton#translateButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 10px;
            }
            QPushButton#translateButton:hover {
                background-color: #2563eb;
            }
            QTextEdit, QTreeWidget {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                font-family: Arial;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: none;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #f1f5f9;
                padding: 10px 16px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #3b82f6;
                color: white;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #e2e8f0;
                height: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 16px;
                padding: 12px;
            }
            QGroupBox::title {
                background-color: transparent;
                padding: 8px;
                color: #1e293b;
                font-weight: bold;
            }
            QPushButton#fileButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                min-width: 150px;
            }
            QPushButton#fileButton:hover {
                background-color: #2563eb;
            }
        """)

        # Central widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header with logo and title
        header = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setFixedSize(32, 32)
        logo_label.setStyleSheet("background-color: #3b82f6; border-radius: 16px;")
        title_label = QLabel("eSRT")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b;")
        header.addWidget(logo_label)
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)

        # Create stacked widget for managing upload and analysis views
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # In the init_ui method, update the upload page section:

        # Create upload page
        upload_page = QWidget()
        upload_layout = QVBoxLayout(upload_page)
        upload_layout.setContentsMargins(40, 40, 40, 40)

        # Add title and subtitle
        title = QLabel("Upload your Document")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #1e293b;
            margin-bottom: 8px;
        """)
        subtitle = QLabel("(.pdf Files only)")
        subtitle.setStyleSheet("color: #64748b; font-size: 14px;")

        upload_layout.addWidget(title, alignment=Qt.AlignCenter)
        upload_layout.addWidget(subtitle, alignment=Qt.AlignCenter)
        upload_layout.addSpacing(40)  # Add more space before button

        # Simple upload button
        self.file_btn = QPushButton('Upload Document')
        self.file_btn.setObjectName("fileButton")
        self.file_btn.clicked.connect(self.select_file)
        self.file_btn.setMinimumWidth(200)  # Make button wider
        self.file_btn.setMinimumHeight(50)  # Make button taller
        upload_layout.addWidget(self.file_btn, alignment=Qt.AlignCenter)

        upload_layout.addStretch()  # Add stretch to push everything to top

        # Add pages to stacked widget
        self.stacked_widget.addWidget(upload_page)
        
        # Analysis page with tabs
        analysis_page = QWidget()
        analysis_layout = QVBoxLayout(analysis_page)
        
        # Tabs
        self.tabs = QTabWidget()
        self.setup_analysis_tabs()
        analysis_layout.addWidget(self.tabs)

        # Add pages to stacked widget
        self.stacked_widget.addWidget(upload_page)
        self.stacked_widget.addWidget(analysis_page)

        # Start with upload page
        self.stacked_widget.setCurrentIndex(0)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: white;
                color: #64748b;
            }
        """)
        self.statusBar().showMessage("Ready")

    def open_pdf(self, pdf_path: str):
        """Open PDF file using system default PDF viewer"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', pdf_path])
            elif platform.system() == 'Windows':
                os.startfile(pdf_path)
            else:  # Linux
                subprocess.run(['xdg-open', pdf_path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening PDF: {str(e)}")

    def setup_analysis_tabs(self):
        # Analysis Results tab containing all elements
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Entities and Statutes
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Entities
        entities_group = QGroupBox("Extracted Entities")
        entities_layout = QVBoxLayout(entities_group)
        self.entities_tree = QTreeWidget()
        self.entities_tree.setHeaderLabel("Entities")
        entities_layout.addWidget(self.entities_tree)
        left_layout.addWidget(entities_group)
        
        # Statutes
        statutes_group = QGroupBox("Statutes")
        statutes_layout = QVBoxLayout(statutes_group)
        self.statutes_tree = QTreeWidget()
        self.statutes_tree.setHeaderLabel("Extracted Statutes")
        statutes_layout.addWidget(self.statutes_tree)
        left_layout.addWidget(statutes_group)
        
        # Right side - Summary and Translation
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Summary
        summary_group = QGroupBox("Case Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        # Translation button
        self.translate_btn = QPushButton('ಕನ್ನಡ ಅನುವಾದ')
        self.translate_btn.setObjectName("translateButton")
        self.translate_btn.clicked.connect(self.translate_summary)
        summary_layout.addWidget(self.translate_btn)
        
        right_layout.addWidget(summary_group)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([400, 600])
        
        analysis_layout.addWidget(splitter)
        self.tabs.addTab(analysis_widget, "Analysis Results")
        
        # Translated text tab
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        self.translated_text.setFont(QFont("Arial", 12))
        self.tabs.addTab(self.translated_text, "ಕನ್ನಡ ಅನುವಾದ")

        # Similar Cases tab
        similar_cases_widget = QWidget()
        similar_cases_layout = QVBoxLayout(similar_cases_widget)

        self.similar_cases_tree = QTreeWidget()
        self.similar_cases_tree.setHeaderLabels(["Similar Cases", "Score", "Matches", "Actions"])
        self.similar_cases_tree.setColumnWidth(0, 300)
        self.similar_cases_tree.setColumnWidth(1, 100)
        self.similar_cases_tree.setColumnWidth(2, 100)
        self.similar_cases_tree.setColumnWidth(3, 100)
        self.similar_cases_tree.setAlternatingRowColors(True)
        
        # Add double-click handler
        self.similar_cases_tree.itemDoubleClicked.connect(self.handle_case_click)
        similar_cases_layout.addWidget(self.similar_cases_tree)

        self.tabs.addTab(similar_cases_widget, "Similar Cases")

    def handle_case_click(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on similar case item"""
        if not item.parent():  # Only handle clicks on main case items
            case_id = item.text(0)
            pdf_path = self.processor.db_manager.get_pdf_path(case_id)
            
            if pdf_path:
                self.open_pdf(pdf_path)
            else:
                QMessageBox.warning(self, "Warning", "PDF file not found for this case.")

    def handle_results(self, results):
        self.progress_bar.hide()
        self.file_btn.setEnabled(True)
        self.statusBar().showMessage("Document Processed Successfully")
        
        # Switch to analysis page
        self.stacked_widget.setCurrentIndex(1)

        # Clear previous results
        self.entities_tree.clear()
        self.summary_text.clear()
        self.statutes_tree.clear()
        self.translated_text.clear()
        self.current_summary = ""
        self.similar_cases_tree.clear()

        similar_cases = results.get('similar_cases', [])
        if similar_cases:
            for case in similar_cases:
                item = QTreeWidgetItem(self.similar_cases_tree)
                item.setText(0, f"{case['case_id']}")
                item.setText(1, f"{case['similarity_score']:.2f}")
                item.setText(2, str(case['total_matches']))
                
                # Add view button indicator
                pdf_path = self.processor.db_manager.get_pdf_path(case['case_id'])
                item.setText(3, "Double-click to view" if pdf_path else "PDF not available")
                
                # Add matching references
                for category, refs in case['matching_references'].items():
                    ref_item = QTreeWidgetItem(item)
                    ref_item.setText(0, f"{category} ({len(refs)} matches)")
                    
                    # Add individual references
                    for ref in refs:
                        match_item = QTreeWidgetItem(ref_item)
                        match_item.setText(0, ref)
                        
            self.similar_cases_tree.expandAll()
        else:
            item = QTreeWidgetItem(self.similar_cases_tree)
            item.setText(0, "No similar cases found")

        # Update entities tree
        for entity_type, entities in results['entities'].items():
            if entity_type != 'STATUTES':
                parent = QTreeWidgetItem(self.entities_tree, [entity_type])
                for entity in entities:
                    QTreeWidgetItem(parent, [entity])
        self.entities_tree.expandAll()

        # Update statutes tree
        if 'STATUTES' in results['entities']:
            for statute in results['entities']['STATUTES']:
                QTreeWidgetItem(self.statutes_tree, [statute])

        # Update summary
        summary_text = "JUDGMENT SUMMARY:\n"
        summary_text += "=" * 50 + "\n\n"
        if results['sections']['judgment_summary']:
            summary_text += results['sections']['judgment_summary']
        else:
            summary_text += "No judgment summary available.\n"
        
        summary_text += "\n\nORDER SUMMARY:\n"
        summary_text += "=" * 50 + "\n\n"
        if results['sections']['order_summary']:
            summary_text += results['sections']['order_summary']
        else:
            summary_text += "No order summary available.\n"
                
        self.summary_text.setText(summary_text)
        self.current_summary = summary_text

    def translate_summary(self):
        if not self.current_summary:
            QMessageBox.warning(self, "Warning", "No Summary Available to Translate")
            return
                
        self.translate_btn.setEnabled(False)
        self.statusBar().showMessage("Translating...")
        
        self.translation_thread = TranslationThread(self.current_summary)
        self.translation_thread.finished.connect(self.handle_translation)
        self.translation_thread.error.connect(self.handle_translation_error)
        self.translation_thread.start()

    def handle_translation(self, translated_text):
        self.translated_text.setText(translated_text)
        self.tabs.setCurrentWidget(self.translated_text)
        self.translate_btn.setEnabled(True)
        self.statusBar().showMessage("Translation completed")

    def handle_translation_error(self, error_message):
        self.translate_btn.setEnabled(True)
        self.statusBar().showMessage("Translation Error")
        QMessageBox.critical(self, "Error", f"Error translating text: {error_message}")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF Document", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.process_document(file_path)

    def process_document(self, file_path):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.file_btn.setEnabled(False)
        self.statusBar().showMessage("Processing document...")

        self.thread = ProcessingThread(self.processor, file_path)
        self.thread.finished.connect(self.handle_results)
        self.thread.error.connect(self.handle_error)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()

    def update_progress(self, message):
        self.statusBar().showMessage(message)

    def handle_error(self, error_message):
        self.progress_bar.hide()
        self.file_btn.setEnabled(True)
        self.statusBar().showMessage("Error occurred")
        QMessageBox.critical(self, "Error", f"Error Processing Document: {error_message}")

def create_application(processor):
    app = QApplication(sys.argv)
    window = LegalAnalyzerGUI(processor)
    window.show()
    return app, window