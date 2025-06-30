"""
Test suite for Writer Agent - Tests proposal generation with persona conditioning.

This module provides comprehensive tests for the Writer Agent including:
- Persona loading and validation
- Section prompt handling
- Content generation (mocked for CI)
- Output format validation
- Token usage tracking
"""

import os
import json
import uuid
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

# Import directly to avoid package init issues
import sys
sys.path.insert(0, 'backend/agents')
from writer_agent import WriterAgent, WriterInput, WriterOutput, Section


class TestWriterAgent:
    """Test cases for Writer Agent functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_personas(self, temp_dir):
        """Create mock personas configuration."""
        personas_data = {
            "personas": {
                "technical": {
                    "name": "Technical Expert",
                    "description": "Deep technical knowledge",
                    "writing_style": {
                        "tone": "analytical",
                        "perspective": "technical",
                        "language_level": "expert",
                        "focus_areas": ["technical_feasibility", "architecture"]
                    },
                    "prompt_additions": "Write with technical precision and expertise."
                },
                "sales": {
                    "name": "Sales Professional",
                    "description": "Persuasive and client-focused",
                    "writing_style": {
                        "tone": "persuasive",
                        "perspective": "client_focused",
                        "language_level": "accessible",
                        "focus_areas": ["value_proposition", "benefits"]
                    },
                    "prompt_additions": "Write with persuasive skill and client focus."
                }
            },
            "default_persona": "technical"
        }
        
        personas_file = temp_dir / "personas.json"
        with open(personas_file, 'w') as f:
            json.dump(personas_data, f)
        
        return personas_file
    
    @pytest.fixture
    def mock_section_prompts(self, temp_dir):
        """Create mock section prompt templates."""
        section_prompts_dir = temp_dir / "section_prompts"
        section_prompts_dir.mkdir()
        
        # Executive summary prompt
        exec_prompt = """## Executive Summary Section Prompt
        
You are writing the Executive Summary section of a proposal.
Provide a compelling overview of the proposed solution."""
        
        with open(section_prompts_dir / "executive_summary.txt", 'w') as f:
            f.write(exec_prompt)
        
        # Technical approach prompt
        tech_prompt = """## Technical Approach Section Prompt
        
You are writing the Technical Approach section of a proposal.
Detail the technical solution and architecture."""
        
        with open(section_prompts_dir / "technical_approach.txt", 'w') as f:
            f.write(tech_prompt)
        
        return section_prompts_dir
    
    @pytest.fixture
    def mock_retrieval_context(self):
        """Create mock retrieval context data."""
        return {
            "id": "test-retrieval-123",
            "matches": [
                {
                    "text": "Previous project implemented web authentication using OAuth2 and JWT tokens.",
                    "score": 0.95,
                    "metadata": {
                        "source": "project_auth.pdf",
                        "chunk_id": "chunk_1"
                    }
                },
                {
                    "text": "Database design used PostgreSQL with proper indexing for performance.",
                    "score": 0.87,
                    "metadata": {
                        "source": "project_db.pdf",
                        "chunk_id": "chunk_2"
                    }
                }
            ]
        }
    
    def test_initialization(self, temp_dir, mock_personas, mock_section_prompts):
        """Test Writer Agent initialization."""
        with patch('backend.agents.writer_agent.genai') as mock_genai:
            mock_genai.configure = Mock()
            mock_genai.GenerativeModel = Mock()
            
            agent = WriterAgent(
                personas_path=str(mock_personas),
                section_prompts_dir=str(mock_section_prompts),
                logs_dir=str(temp_dir / "logs")
            )
            
            assert agent.personas_path == mock_personas
            assert agent.section_prompts_dir == mock_section_prompts
            assert len(agent.personas["personas"]) == 2
            assert len(agent.section_prompts) == 2
            assert "executive_summary" in agent.section_prompts
            assert "technical_approach" in agent.section_prompts
    
    def test_persona_loading(self, temp_dir, mock_personas):
        """Test persona configuration loading."""
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                personas_path=str(mock_personas),
                logs_dir=str(temp_dir / "logs")
            )
            
            # Check personas loaded correctly
            assert "technical" in agent.personas["personas"]
            assert "sales" in agent.personas["personas"]
            assert agent.personas["default_persona"] == "technical"
            
            tech_persona = agent.personas["personas"]["technical"]
            assert tech_persona["name"] == "Technical Expert"
            assert "technical_feasibility" in tech_persona["writing_style"]["focus_areas"]
    
    def test_section_prompts_loading(self, temp_dir, mock_section_prompts):
        """Test section prompt template loading."""
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                section_prompts_dir=str(mock_section_prompts),
                logs_dir=str(temp_dir / "logs")
            )
            
            assert "executive_summary" in agent.section_prompts
            assert "technical_approach" in agent.section_prompts
            assert "Executive Summary Section Prompt" in agent.section_prompts["executive_summary"]
            assert "Technical Approach Section Prompt" in agent.section_prompts["technical_approach"]
    
    def test_prompt_construction(self, temp_dir, mock_personas, mock_section_prompts, mock_retrieval_context):
        """Test section prompt construction with persona and context."""
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                personas_path=str(mock_personas),
                section_prompts_dir=str(mock_section_prompts),
                logs_dir=str(temp_dir / "logs")
            )
            
            prompt = agent._construct_section_prompt(
                section_type="executive_summary",
                user_prompt="Build a web application with user authentication",
                persona="technical",
                retrieval_context=mock_retrieval_context
            )
            
            # Check prompt contains expected elements
            assert "Executive Summary Section Prompt" in prompt
            assert "Build a web application with user authentication" in prompt
            assert "Write with technical precision and expertise" in prompt
            assert "analytical tone" in prompt
            assert "OAuth2 and JWT tokens" in prompt
            assert "project_auth.pdf" in prompt
    
    @patch('backend.agents.writer_agent.genai')
    def test_section_content_generation(self, mock_genai, temp_dir, mock_personas, mock_section_prompts):
        """Test individual section content generation."""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "# Executive Summary\n\nThis is a test proposal section."
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        agent = WriterAgent(
            personas_path=str(mock_personas),
            section_prompts_dir=str(mock_section_prompts),
            logs_dir=str(temp_dir / "logs")
        )
        
        result = agent._generate_section_content(
            section_type="executive_summary",
            user_prompt="Build a web application",
            persona="technical"
        )
        
        assert result["section_type"] == "executive_summary"
        assert result["title"] == "Executive Summary"
        assert "markdown" in result["content"]
        assert "html" in result["content"]
        assert result["word_count"] > 0
        assert result["generation_metadata"]["prompt_tokens"] == 100
        assert result["generation_metadata"]["completion_tokens"] == 50
    
    @patch('backend.agents.writer_agent.genai')
    def test_full_proposal_generation(self, mock_genai, temp_dir, mock_personas, mock_section_prompts, mock_retrieval_context):
        """Test complete proposal generation workflow."""
        # Mock Gemini responses for multiple sections
        def mock_generate_content(prompt):
            if "Executive Summary" in prompt:
                content = "# Executive Summary\n\nComprehensive web application solution."
            elif "Technical Approach" in prompt:
                content = "# Technical Approach\n\nReact frontend with Node.js backend."
            else:
                content = "# Section\n\nGeneric section content."
            
            mock_response = Mock()
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].content.parts = [Mock()]
            mock_response.candidates[0].content.parts[0].text = content
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 75
            return mock_response
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = mock_generate_content
        mock_genai.GenerativeModel.return_value = mock_model
        
        agent = WriterAgent(
            personas_path=str(mock_personas),
            section_prompts_dir=str(mock_section_prompts),
            logs_dir=str(temp_dir / "logs")
        )
        
        writer_input = WriterInput(
            user_prompt="Build a web application with user authentication and data analytics",
            persona="technical",
            retrieval_context=mock_retrieval_context,
            sections_to_generate=["executive_summary", "technical_approach"]
        )
        
        result = agent.generate(writer_input)
        
        # Validate output structure
        assert isinstance(result, WriterOutput)
        assert result.generation_id
        assert result.timestamp
        assert len(result.generated_content["sections"]) == 2
        assert result.generated_content["word_count"] > 0
        assert result.generation_metadata["token_usage"]["total_tokens"] > 0
        
        # Check sections
        sections = result.generated_content["sections"]
        assert sections[0]["section_type"] == "executive_summary"
        assert sections[1]["section_type"] == "technical_approach"
        
        # Check full content
        full_content = result.generated_content["full_content"]
        assert "markdown" in full_content
        assert "html" in full_content
        assert "Executive Summary" in full_content["markdown"]
        assert "Technical Approach" in full_content["markdown"]
    
    @patch('backend.agents.writer_agent.genai')
    def test_result_saving(self, mock_genai, temp_dir, mock_personas, mock_section_prompts):
        """Test saving generation results to files."""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "# Test Section\n\nTest content."
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 25
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        agent = WriterAgent(
            personas_path=str(mock_personas),
            section_prompts_dir=str(mock_section_prompts),
            logs_dir=str(temp_dir / "logs")
        )
        
        writer_input = WriterInput(
            user_prompt="Test prompt",
            persona="technical",
            sections_to_generate=["executive_summary"]
        )
        
        result = agent.generate(writer_input)
        
        # Save results
        output_dir = temp_dir / "output"
        saved_path = agent.save_result(result, str(output_dir))
        
        # Check files created
        assert os.path.exists(saved_path)
        assert (output_dir / f"proposal_{result.generation_id}.md").exists()
        assert (output_dir / f"proposal_{result.generation_id}.html").exists()
        
        # Check JSON content
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["generation_id"] == result.generation_id
        assert "generated_content" in saved_data
        assert "generation_metadata" in saved_data
    
    def test_token_usage_logging(self, temp_dir, mock_personas, mock_section_prompts):
        """Test token usage CSV logging."""
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                personas_path=str(mock_personas),
                section_prompts_dir=str(mock_section_prompts),
                logs_dir=str(temp_dir / "logs")
            )
            
            # Log token usage
            generation_id = str(uuid.uuid4())
            agent._log_token_usage(
                generation_id=generation_id,
                model="gemini-2.5-flash",
                persona="technical",
                prompt_tokens=100,
                completion_tokens=50,
                section_type="executive_summary",
                generation_time_ms=1500.0
            )
            
            # Check CSV file created and contains data
            csv_file = temp_dir / "logs" / "token_usage.csv"
            assert csv_file.exists()
            
            with open(csv_file, 'r') as f:
                content = f.read()
                assert generation_id in content
                assert "gemini-2.5-flash" in content
                assert "technical" in content
                assert "100" in content
                assert "50" in content
    
    def test_error_handling(self, temp_dir):
        """Test error handling for missing files and invalid configurations."""
        # Test missing personas file
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                personas_path="nonexistent/personas.json",
                logs_dir=str(temp_dir / "logs")
            )
            
            assert agent.personas["personas"] == {}
            assert agent.personas["default_persona"] == "consultant"
        
        # Test missing section prompts directory
        with patch('backend.agents.writer_agent.genai'):
            agent = WriterAgent(
                section_prompts_dir="nonexistent/prompts",
                logs_dir=str(temp_dir / "logs")
            )
            
            assert agent.section_prompts == {}
    
    def test_writer_input_defaults(self):
        """Test WriterInput dataclass default values."""
        writer_input = WriterInput(user_prompt="Test prompt")
        
        assert writer_input.persona == "consultant"
        assert writer_input.retrieval_context is None
        assert "executive_summary" in writer_input.sections_to_generate
        assert "technical_approach" in writer_input.sections_to_generate
        assert "project_management" in writer_input.sections_to_generate
        assert writer_input.generation_params["temperature"] == 0.7
        assert writer_input.generation_params["max_tokens"] == 4000
    
    def test_section_model_validation(self):
        """Test Section Pydantic model validation."""
        section_data = {
            "section_id": "test-123",
            "section_type": "executive_summary",
            "title": "Executive Summary",
            "content": {
                "markdown": "# Test\n\nContent",
                "html": "<h1>Test</h1><p>Content</p>"
            },
            "word_count": 2,
            "sources_referenced": ["source1.pdf"],
            "confidence_score": 0.9
        }
        
        section = Section(**section_data)
        assert section.section_id == "test-123"
        assert section.section_type == "executive_summary"
        assert section.word_count == 2
        assert len(section.sources_referenced) == 1
        assert section.confidence_score == 0.9


# Integration tests (require actual API keys)
@pytest.mark.integration
class TestWriterAgentIntegration:
    """Integration tests that require real API access."""
    
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set"
    )
    def test_real_gemini_generation(self):
        """Test actual content generation with Gemini API."""
        agent = WriterAgent()
        
        writer_input = WriterInput(
            user_prompt="Create a simple web application for task management",
            persona="technical",
            sections_to_generate=["executive_summary"]
        )
        
        result = agent.generate(writer_input)
        
        assert result.generation_id
        assert result.generated_content["word_count"] > 10
        assert len(result.generated_content["sections"]) == 1
        assert result.generation_metadata["token_usage"]["total_tokens"] > 0
        
        # Save results for manual inspection
        saved_path = agent.save_result(result)
        print(f"Integration test results saved to: {saved_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 