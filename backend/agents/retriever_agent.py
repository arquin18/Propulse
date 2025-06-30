"""
Retriever Agent
Performs semantic search across dual vector databases (RFPs and Proposals).
"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from pydantic import BaseModel, ValidationError

from core.extract_text import TextExtractor, TextChunk


class QueryInput(BaseModel):
    """Input query structure."""
    text: str
    document_path: Optional[str] = None
    top_k: int = 10
    similarity_threshold: float = 0.1


class RetrievalMatch(BaseModel):
    """Single retrieval match."""
    id: int
    content: str
    source_file: str
    similarity_score: float
    chunk_metadata: Dict[str, Any]


class RetrievalResult(BaseModel):
    """Complete retrieval result following MCP schema."""
    retrieval_id: str
    timestamp: str
    query: Dict[str, Any]
    results: Dict[str, Any]
    metadata: Dict[str, Any]


class VectorDatabase:
    """Vector database wrapper for FAISS index and chunks."""
    
    def __init__(self, db_path: Path):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to vector database directory
        """
        self.db_path = Path(db_path)
        self.index = None
        self.chunks = []
        self.metadata = {}
        
        if self.db_path.exists():
            self.load()
    
    def load(self):
        """Load vector database from disk."""
        try:
            # Load FAISS index
            index_path = self.db_path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
            
            # Load chunks
            chunks_path = self.db_path / "chunks.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks")
            
            # Load metadata
            metadata_path = self.db_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded database metadata: {self.metadata.get('document_type', 'unknown')}")
        
        except Exception as e:
            logger.error(f"Error loading vector database from {self.db_path}: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if database is properly loaded."""
        return self.index is not None and len(self.chunks) > 0
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.1
    ) -> List[RetrievalMatch]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval matches
        """
        if not self.is_loaded():
            logger.warning("Vector database not loaded")
            return []
        
        try:
            # Perform search
            scores, indices = self.index.search(
                query_embedding.astype(np.float32).reshape(1, -1),
                min(top_k, self.index.ntotal)
            )
            
            matches = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                if score < similarity_threshold:
                    continue
                
                chunk_data = self.chunks[idx]
                match = RetrievalMatch(
                    id=chunk_data["id"],
                    content=chunk_data["content"],
                    source_file=chunk_data["source_file"],
                    similarity_score=float(score),
                    chunk_metadata=chunk_data["metadata"]
                )
                matches.append(match)
            
            logger.info(f"Found {len(matches)} matches above threshold {similarity_threshold}")
            return matches
        
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []


class RetrieverAgent:
    """
    Retriever Agent for semantic search across dual vector databases.
    """
    
    def __init__(
        self,
        rfp_db_path: str,
        proposal_db_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        log_file: str = "logs/retriever_log.jsonl"
    ):
        """
        Initialize the Retriever Agent.
        
        Args:
            rfp_db_path: Path to RFP vector database
            proposal_db_path: Path to proposal vector database
            model_name: Sentence transformer model name
            log_file: Path to log file
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.text_extractor = TextExtractor()
        
        # Load vector databases
        self.rfp_db = VectorDatabase(Path(rfp_db_path))
        self.proposal_db = VectorDatabase(Path(proposal_db_path))
        
        # Setup logging
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RetrieverAgent with model: {model_name}")
        logger.info(f"RFP DB loaded: {self.rfp_db.is_loaded()}")
        logger.info(f"Proposal DB loaded: {self.proposal_db.is_loaded()}")
    
    def _extract_query_text(self, query: QueryInput) -> str:
        """
        Extract and combine text from query input.
        
        Args:
            query: Input query
            
        Returns:
            Combined query text
        """
        query_text = query.text or ""
        
        # If document is provided, extract text and combine
        if query.document_path:
            try:
                doc_path = Path(query.document_path)
                if doc_path.exists():
                    doc_text = self.text_extractor.extract_text(doc_path)
                    query_text = f"{query_text}\n\n{doc_text}".strip()
                    logger.info(f"Combined query with document: {len(doc_text)} characters")
                else:
                    logger.warning(f"Document not found: {query.document_path}")
            except Exception as e:
                logger.error(f"Error extracting document text: {e}")
        
        return query_text
    
    def _embed_query(self, query_text: str) -> np.ndarray:
        """
        Create embedding for query text.
        
        Args:
            query_text: Text to embed
            
        Returns:
            Query embedding
        """
        embedding = self.encoder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def _determine_query_type(self, query: QueryInput) -> str:
        """Determine the type of query based on inputs."""
        has_text = bool(query.text and query.text.strip())
        has_doc = bool(query.document_path)
        
        if has_text and has_doc:
            return "text_and_document"
        elif has_doc:
            return "document_only"
        else:
            return "text_only"
    
    def retrieve(self, query: QueryInput) -> RetrievalResult:
        """
        Perform retrieval across both vector databases.
        
        Args:
            query: Input query
            
        Returns:
            Retrieval result following MCP schema
        """
        start_time = time.time()
        retrieval_id = str(uuid.uuid4())
        
        logger.info(f"Starting retrieval {retrieval_id}")
        
        try:
            # Extract and embed query
            query_text = self._extract_query_text(query)
            if not query_text.strip():
                raise ValueError("No query text provided")
            
            query_embedding = self._embed_query(query_text)
            
            # Search both databases
            rfp_matches = []
            proposal_matches = []
            
            if self.rfp_db.is_loaded():
                rfp_matches = self.rfp_db.search(
                    query_embedding,
                    top_k=query.top_k,
                    similarity_threshold=query.similarity_threshold
                )
            
            if self.proposal_db.is_loaded():
                proposal_matches = self.proposal_db.search(
                    query_embedding,
                    top_k=query.top_k,
                    similarity_threshold=query.similarity_threshold
                )
            
            # Calculate retrieval time
            retrieval_time = (time.time() - start_time) * 1000
            
            # Create result
            result = RetrievalResult(
                retrieval_id=retrieval_id,
                timestamp=datetime.now().isoformat(),
                query={
                    "text": query.text or "",
                    "document_path": query.document_path,
                    "query_type": self._determine_query_type(query)
                },
                results={
                    "rfp_matches": [match.dict() for match in rfp_matches],
                    "proposal_matches": [match.dict() for match in proposal_matches],
                    "total_matches": len(rfp_matches) + len(proposal_matches)
                },
                metadata={
                    "retrieval_time_ms": retrieval_time,
                    "model_used": self.model_name,
                    "search_parameters": {
                        "top_k": query.top_k,
                        "similarity_threshold": query.similarity_threshold
                    }
                }
            )
            
            # Log the retrieval
            self._log_retrieval(result, rfp_matches, proposal_matches)
            
            logger.info(f"Retrieval {retrieval_id} completed in {retrieval_time:.2f}ms")
            logger.info(f"Found {len(rfp_matches)} RFP matches, {len(proposal_matches)} proposal matches")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during retrieval {retrieval_id}: {e}")
            # Return empty result with error
            return RetrievalResult(
                retrieval_id=retrieval_id,
                timestamp=datetime.now().isoformat(),
                query={
                    "text": query.text or "",
                    "document_path": query.document_path,
                    "query_type": self._determine_query_type(query)
                },
                results={
                    "rfp_matches": [],
                    "proposal_matches": [],
                    "total_matches": 0
                },
                metadata={
                    "retrieval_time_ms": (time.time() - start_time) * 1000,
                    "model_used": self.model_name,
                    "search_parameters": {
                        "top_k": query.top_k,
                        "similarity_threshold": query.similarity_threshold
                    },
                    "error": str(e)
                }
            )
    
    def _log_retrieval(
        self,
        result: RetrievalResult,
        rfp_matches: List[RetrievalMatch],
        proposal_matches: List[RetrievalMatch]
    ):
        """Log retrieval metadata to JSONL file."""
        try:
            log_entry = {
                "retrieval_id": result.retrieval_id,
                "timestamp": result.timestamp,
                "query_length": len(result.query.get("text", "")),
                "retrieval_time_ms": result.metadata["retrieval_time_ms"],
                "rfp_matches_count": len(rfp_matches),
                "proposal_matches_count": len(proposal_matches),
                "top_rfp_score": max([m.similarity_score for m in rfp_matches], default=0.0),
                "top_proposal_score": max([m.similarity_score for m in proposal_matches], default=0.0),
                "rfp_source_files": list(set([m.source_file for m in rfp_matches])),
                "proposal_source_files": list(set([m.source_file for m in proposal_matches])),
                "model_used": result.metadata["model_used"]
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        except Exception as e:
            logger.error(f"Error logging retrieval: {e}")
    
    def save_result(self, result: RetrievalResult, output_path: Optional[str] = None):
        """
        Save retrieval result to file.
        
        Args:
            result: Retrieval result
            output_path: Output file path (optional)
        """
        if not output_path:
            output_path = f"shared/mcp_schemas/retriever_output_{result.retrieval_id}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved retrieval result to {output_file}")


def main():
    """Example usage of the Retriever Agent."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize agent
    agent = RetrieverAgent(
        rfp_db_path=os.getenv("VECTOR_DB_RFP_PATH", "data/vector_dbs/rfp_db"),
        proposal_db_path=os.getenv("VECTOR_DB_PROPOSAL_PATH", "data/vector_dbs/proposal_db")
    )
    
    # Example query
    query = QueryInput(
        text="We need a web application for project management with user authentication",
        top_k=5,
        similarity_threshold=0.2
    )
    
    # Perform retrieval
    result = agent.retrieve(query)
    
    # Save result
    agent.save_result(result)
    
    print(f"Retrieval completed: {result.results['total_matches']} total matches found")


if __name__ == "__main__":
    main() 