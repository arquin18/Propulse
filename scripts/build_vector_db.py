"""
Vector Database Builder
Builds FAISS vector databases from document collections.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import sys
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import os

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))
from core.extract_text import TextExtractor, TextChunk


class VectorDBBuilder:
    """Build and manage FAISS vector databases."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384
    ):
        """
        Initialize the vector database builder.
        
        Args:
            model_name: Name of the sentence transformer model
            dimension: Embedding dimension
        """
        self.model_name = model_name
        self.dimension = dimension
        self.encoder = SentenceTransformer(model_name)
        self.text_extractor = TextExtractor()
        
        logger.info(f"Initialized VectorDBBuilder with model: {model_name}")
    
    def process_documents(
        self,
        doc_dir: Path,
        doc_type: str = "rfp"
    ) -> List[TextChunk]:
        """
        Process all documents in a directory.
        
        Args:
            doc_dir: Directory containing documents
            doc_type: Type of documents (rfp, proposal, etc.)
            
        Returns:
            List of text chunks from all documents
        """
        doc_dir = Path(doc_dir)
        if not doc_dir.exists():
            logger.warning(f"Directory does not exist: {doc_dir}")
            return []
        
        all_chunks = []
        supported_extensions = {'.pdf', '.docx', '.txt'}
        
        for file_path in doc_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    metadata = {
                        "document_type": doc_type,
                        "category": file_path.parent.name,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    chunks = self.text_extractor.extract_and_chunk(
                        file_path, 
                        metadata
                    )
                    all_chunks.extend(chunks)
                    
                    logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def create_embeddings(self, chunks: List[TextChunk]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        if not chunks:
            return np.array([]).reshape(0, self.dimension)
        
        texts = [chunk.content for chunk in chunks]
        
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        use_gpu: bool = False
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            FAISS index
        """
        if embeddings.shape[0] == 0:
            # Create empty index
            index = faiss.IndexFlatIP(self.dimension)
            return index
        
        # Use Inner Product for cosine similarity (with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS index")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(embeddings.astype(np.float32))
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index
    
    def save_vector_db(
        self,
        index: faiss.Index,
        chunks: List[TextChunk],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save vector database to disk.
        
        Args:
            index: FAISS index
            chunks: Text chunks
            output_path: Output directory path
            metadata: Additional metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_path / "index.faiss"
        faiss.write_index(faiss.index_gpu_to_cpu(index) if hasattr(index, 'device') else index, str(index_path))
        
        # Save chunks metadata
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": i,
                "content": chunk.content,
                "source_file": chunk.source_file,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata
            }
            chunks_data.append(chunk_data)
        
        chunks_path = output_path / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save database metadata
        db_metadata = {
            "created_at": datetime.now().isoformat(),
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_chunks": len(chunks),
            "total_documents": len(set(chunk.source_file for chunk in chunks)),
            **(metadata or {})
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(db_metadata, f, indent=2)
        
        logger.info(f"Saved vector database to {output_path}")
        logger.info(f"  - Index: {index_path}")
        logger.info(f"  - Chunks: {chunks_path}")
        logger.info(f"  - Metadata: {metadata_path}")
    
    def build_database(
        self,
        doc_dir: Path,
        output_path: Path,
        doc_type: str = "documents",
        use_gpu: bool = False
    ):
        """
        Build complete vector database from documents.
        
        Args:
            doc_dir: Directory containing documents
            output_path: Output path for vector database
            doc_type: Type of documents
            use_gpu: Whether to use GPU acceleration
        """
        logger.info(f"Building vector database from {doc_dir}")
        
        # Process documents
        chunks = self.process_documents(doc_dir, doc_type)
        
        if not chunks:
            logger.warning("No chunks found. Creating empty database.")
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings, use_gpu)
        
        # Save database
        metadata = {
            "source_directory": str(doc_dir),
            "document_type": doc_type
        }
        self.save_vector_db(index, chunks, output_path, metadata)
        
        logger.info(f"Vector database build complete: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Build vector databases from documents")
    
    parser.add_argument(
        "--rfp-dir",
        type=Path,
        default=Path("shared/sample_rfps"),
        help="Directory containing RFP documents"
    )
    
    parser.add_argument(
        "--proposal-dir", 
        type=Path,
        default=Path("shared/templates"),
        help="Directory containing proposal templates"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path, 
        default=Path("data/vector_dbs"),
        help="Output directory for vector databases"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        "logs/vector_db_build.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    builder = VectorDBBuilder(model_name=args.model)
    
    # Build RFP database
    if args.rfp_dir.exists():
        rfp_output = args.output_dir / "rfp_db"
        logger.info("Building RFP vector database...")
        builder.build_database(
            args.rfp_dir,
            rfp_output,
            doc_type="rfp",
            use_gpu=args.gpu
        )
    else:
        logger.warning(f"RFP directory not found: {args.rfp_dir}")
    
    # Build proposal database
    if args.proposal_dir.exists():
        proposal_output = args.output_dir / "proposal_db"
        logger.info("Building proposal vector database...")
        builder.build_database(
            args.proposal_dir,
            proposal_output,
            doc_type="proposal",
            use_gpu=args.gpu
        )
    else:
        logger.warning(f"Proposal directory not found: {args.proposal_dir}")
    
    logger.info("Vector database building completed!")


if __name__ == "__main__":
    main() 