"""
Tests for Retriever Agent
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from agents.retriever_agent import (
    RetrieverAgent, 
    VectorDatabase, 
    QueryInput, 
    RetrievalMatch,
    RetrievalResult
)
from core.extract_text import TextExtractor, TextChunk


class TestTextExtractor:
    """Test the TextExtractor class."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        extractor = TextExtractor()
        
        # Test basic cleaning
        dirty_text = "  This   is\n\n  some  text  with\textra\r\nwhitespace  "
        clean_text = extractor._clean_text(dirty_text)
        assert clean_text == "This is some text with extra whitespace"
        
        # Test special character removal
        special_text = "Text with special chars: ™ © ® § ¶ and more"
        clean_special = extractor._clean_text(special_text)
        assert "™" not in clean_special
        assert "Text with special chars" in clean_special
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        extractor = TextExtractor(chunk_size=50, chunk_overlap=10, min_chunk_size=20)
        
        text = "This is a sample text for testing chunking. " * 10
        chunks = extractor.chunk_text(text, "test.txt")
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(len(chunk.content) >= 20 for chunk in chunks)
        assert chunks[0].source_file == "test.txt"
    
    @patch('pypdf.PdfReader')
    def test_extract_from_pdf(self, mock_pdf_reader):
        """Test PDF text extraction."""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        extractor = TextExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            text = extractor.extract_from_pdf(Path(tmp_file.name))
            assert "Sample PDF text content" in text
    
    @patch('docx.Document')
    def test_extract_from_docx(self, mock_document):
        """Test DOCX text extraction."""
        # Mock document structure
        mock_para = Mock()
        mock_para.text = "Sample paragraph text"
        mock_document.return_value.paragraphs = [mock_para]
        mock_document.return_value.tables = []
        
        extractor = TextExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            text = extractor.extract_from_docx(Path(tmp_file.name))
            assert "Sample paragraph text" in text


class TestVectorDatabase:
    """Test the VectorDatabase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock database files
        self.mock_chunks = [
            {
                "id": 0,
                "content": "Sample chunk content about web development",
                "source_file": "sample.txt",
                "chunk_id": 0,
                "start_char": 0,
                "end_char": 50,
                "metadata": {
                    "document_type": "rfp",
                    "category": "tech"
                }
            }
        ]
        
        self.mock_metadata = {
            "created_at": "2024-01-01T00:00:00",
            "model_name": "test-model",
            "dimension": 384,
            "total_chunks": 1,
            "document_type": "rfp"
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('faiss.read_index')
    def test_load_database(self, mock_read_index):
        """Test loading vector database."""
        # Create mock files
        chunks_file = self.temp_dir / "chunks.json"
        metadata_file = self.temp_dir / "metadata.json"
        index_file = self.temp_dir / "index.faiss"
        
        with open(chunks_file, 'w') as f:
            json.dump(self.mock_chunks, f)
        
        with open(metadata_file, 'w') as f:
            json.dump(self.mock_metadata, f)
        
        # Create empty index file
        index_file.touch()
        
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 1
        mock_read_index.return_value = mock_index
        
        # Test loading
        db = VectorDatabase(self.temp_dir)
        
        assert db.is_loaded()
        assert len(db.chunks) == 1
        assert db.metadata["document_type"] == "rfp"
        mock_read_index.assert_called_once()
    
    def test_empty_database(self):
        """Test handling of empty/non-existent database."""
        empty_dir = self.temp_dir / "empty"
        db = VectorDatabase(empty_dir)
        
        assert not db.is_loaded()
        assert len(db.chunks) == 0


class TestRetrieverAgent:
    """Test the RetrieverAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.rfp_db_path = self.temp_dir / "rfp_db"
        self.proposal_db_path = self.temp_dir / "proposal_db"
        
        # Create directories
        self.rfp_db_path.mkdir(parents=True)
        self.proposal_db_path.mkdir(parents=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('agents.retriever_agent.VectorDatabase')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retriever_agent_init(self, mock_transformer, mock_vector_db):
        """Test RetrieverAgent initialization."""
        mock_transformer.return_value = Mock()
        mock_db_instance = Mock()
        mock_db_instance.is_loaded.return_value = True
        mock_vector_db.return_value = mock_db_instance
        
        agent = RetrieverAgent(
            rfp_db_path=str(self.rfp_db_path),
            proposal_db_path=str(self.proposal_db_path)
        )
        
        assert agent.model_name == "all-MiniLM-L6-v2"
        assert agent.rfp_db is not None
        assert agent.proposal_db is not None
    
    @patch('agents.retriever_agent.VectorDatabase')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_text_only(self, mock_transformer, mock_vector_db):
        """Test retrieval with text-only query."""
        # Mock transformer
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_encoder
        
        # Mock vector databases
        mock_db_instance = Mock()
        mock_db_instance.is_loaded.return_value = True
        
        mock_match = RetrievalMatch(
            id=0,
            content="Sample matching content",
            source_file="test.txt",
            similarity_score=0.8,
            chunk_metadata={"document_type": "rfp"}
        )
        
        mock_db_instance.search.return_value = [mock_match]
        mock_vector_db.return_value = mock_db_instance
        
        # Create agent
        agent = RetrieverAgent(
            rfp_db_path=str(self.rfp_db_path),
            proposal_db_path=str(self.proposal_db_path)
        )
        
        # Test retrieval
        query = QueryInput(text="test query", top_k=5)
        result = agent.retrieve(query)
        
        assert isinstance(result, RetrievalResult)
        assert result.query["query_type"] == "text_only"
        assert result.results["total_matches"] == 2  # RFP + proposal matches
        assert result.metadata["model_used"] == "all-MiniLM-L6-v2"
    
    @patch('agents.retriever_agent.VectorDatabase')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_with_document(self, mock_transformer, mock_vector_db):
        """Test retrieval with document upload."""
        # Create a test document
        test_doc = self.temp_dir / "test.txt"
        test_doc.write_text("Sample document content for testing")
        
        # Mock transformer
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_encoder
        
        # Mock vector databases
        mock_db_instance = Mock()
        mock_db_instance.is_loaded.return_value = True
        mock_db_instance.search.return_value = []
        mock_vector_db.return_value = mock_db_instance
        
        # Create agent
        agent = RetrieverAgent(
            rfp_db_path=str(self.rfp_db_path),
            proposal_db_path=str(self.proposal_db_path)
        )
        
        # Test retrieval with document
        query = QueryInput(
            text="test query",
            document_path=str(test_doc),
            top_k=5
        )
        result = agent.retrieve(query)
        
        assert result.query["query_type"] == "text_and_document"
        assert result.query["document_path"] == str(test_doc)
    
    @patch('agents.retriever_agent.VectorDatabase')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_error_handling(self, mock_transformer, mock_vector_db):
        """Test error handling in retrieval."""
        # Mock transformer to raise exception
        mock_encoder = Mock()
        mock_encoder.encode.side_effect = Exception("Embedding error")
        mock_transformer.return_value = mock_encoder
        
        # Mock vector databases
        mock_db_instance = Mock()
        mock_db_instance.is_loaded.return_value = True
        mock_vector_db.return_value = mock_db_instance
        
        # Create agent
        agent = RetrieverAgent(
            rfp_db_path=str(self.rfp_db_path),
            proposal_db_path=str(self.proposal_db_path)
        )
        
        # Test retrieval with error
        query = QueryInput(text="test query")
        result = agent.retrieve(query)
        
        assert result.results["total_matches"] == 0
        assert "error" in result.metadata
    
    def test_query_type_determination(self):
        """Test query type determination logic."""
        # Mock dependencies to avoid actual initialization
        with patch('agents.retriever_agent.VectorDatabase'), \
             patch('sentence_transformers.SentenceTransformer'):
            
            agent = RetrieverAgent(
                rfp_db_path=str(self.rfp_db_path),
                proposal_db_path=str(self.proposal_db_path)
            )
            
            # Test different query types
            text_only = QueryInput(text="test")
            assert agent._determine_query_type(text_only) == "text_only"
            
            doc_only = QueryInput(text="", document_path="/path/to/doc")
            assert agent._determine_query_type(doc_only) == "document_only"
            
            both = QueryInput(text="test", document_path="/path/to/doc")
            assert agent._determine_query_type(both) == "text_and_document"


class TestQueryInput:
    """Test the QueryInput model."""
    
    def test_valid_query_input(self):
        """Test valid query input creation."""
        query = QueryInput(
            text="Sample query text",
            top_k=5,
            similarity_threshold=0.2
        )
        
        assert query.text == "Sample query text"
        assert query.top_k == 5
        assert query.similarity_threshold == 0.2
        assert query.document_path is None
    
    def test_query_input_with_document(self):
        """Test query input with document path."""
        query = QueryInput(
            text="Query with document",
            document_path="/path/to/document.pdf",
            top_k=10
        )
        
        assert query.document_path == "/path/to/document.pdf"


class TestRetrievalMatch:
    """Test the RetrievalMatch model."""
    
    def test_retrieval_match_creation(self):
        """Test creating a retrieval match."""
        match = RetrievalMatch(
            id=1,
            content="Sample content",
            source_file="test.txt",
            similarity_score=0.85,
            chunk_metadata={"type": "rfp"}
        )
        
        assert match.id == 1
        assert match.similarity_score == 0.85
        assert match.chunk_metadata["type"] == "rfp"


# Integration tests
class TestRetrieverIntegration:
    """Integration tests for the complete retrieval pipeline."""
    
    @pytest.mark.integration
    def test_end_to_end_retrieval(self):
        """Test complete retrieval pipeline (requires actual models)."""
        # This test would require actual sentence transformers and FAISS
        # Skip in unit tests, run only in integration test suite
        pytest.skip("Integration test - requires actual models")


if __name__ == "__main__":
    pytest.main([__file__]) 