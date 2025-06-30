"""
Writer Agent for Propulse - Generates proposals using Gemini with persona conditioning.

This module implements the Writer Agent that takes retrieval context and user prompts,
applies persona-based conditioning, and generates structured proposal content using
Google's Gemini model via the Google Agent Development Kit.
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import csv

import markdown
from pydantic import BaseModel, Field

# Import Google ADK components
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    logging.warning("Google GenerativeAI not available. Install with: pip install google-generativeai")
    genai = None


@dataclass
class WriterInput:
    """Input structure for Writer Agent following MCP protocol."""
    user_prompt: str
    persona: str = "consultant"
    retrieval_context: Optional[Dict[str, Any]] = None
    sections_to_generate: List[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.sections_to_generate is None:
            self.sections_to_generate = ["executive_summary", "technical_approach", "project_management"]
        
        if self.generation_params is None:
            self.generation_params = {
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 0.9
            }


class Section(BaseModel):
    """Represents a proposal section."""
    section_id: str
    section_type: str
    title: str
    content: Dict[str, str]  # markdown and html
    word_count: int
    sources_referenced: List[str] = []
    confidence_score: Optional[float] = None


class WriterOutput(BaseModel):
    """Output structure for Writer Agent following MCP protocol."""
    generation_id: str
    timestamp: str
    input_context: Dict[str, Any]
    generated_content: Dict[str, Any]
    generation_metadata: Dict[str, Any]


class WriterAgent:
    """
    Writer Agent that generates proposal content using Gemini with persona conditioning.
    
    Features:
    - Persona-based content generation
    - Section-specific prompting
    - Markdown to HTML conversion
    - Token usage tracking
    - Comprehensive logging
    """
    
    def __init__(
        self,
        personas_path: str = "shared/personas.json",
        section_prompts_dir: str = "shared/templates/section_prompts",
        model_name: str = "gemini-2.5-flash",
        logs_dir: str = "logs"
    ):
        """
        Initialize the Writer Agent.
        
        Args:
            personas_path: Path to personas configuration file
            section_prompts_dir: Directory containing section prompt templates
            model_name: Gemini model to use
            logs_dir: Directory for log files
        """
        self.personas_path = Path(personas_path)
        self.section_prompts_dir = Path(section_prompts_dir)
        self.model_name = model_name
        self.logs_dir = Path(logs_dir)
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Load personas and section prompts
        self.personas = self._load_personas()
        self.section_prompts = self._load_section_prompts()
        
        # Initialize Gemini
        self.model = self._initialize_gemini()
        
        # Token usage CSV file
        self.token_csv_path = self.logs_dir / "token_usage.csv"
        self._initialize_token_csv()
        
        self.logger.info(f"Writer Agent initialized with model: {self.model_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("writer_agent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.logs_dir / "writer_agent.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_personas(self) -> Dict[str, Any]:
        """Load personas configuration."""
        try:
            with open(self.personas_path, 'r', encoding='utf-8') as f:
                personas_data = json.load(f)
            
            self.logger.info(f"Loaded {len(personas_data.get('personas', {}))} personas")
            return personas_data
        
        except FileNotFoundError:
            self.logger.warning(f"Personas file not found: {self.personas_path}")
            return {"personas": {}, "default_persona": "consultant"}
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in personas file: {e}")
            return {"personas": {}, "default_persona": "consultant"}
    
    def _load_section_prompts(self) -> Dict[str, str]:
        """Load section prompt templates."""
        section_prompts = {}
        
        if not self.section_prompts_dir.exists():
            self.logger.warning(f"Section prompts directory not found: {self.section_prompts_dir}")
            return section_prompts
        
        for prompt_file in self.section_prompts_dir.glob("*.txt"):
            section_name = prompt_file.stem
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    section_prompts[section_name] = f.read()
                self.logger.debug(f"Loaded section prompt: {section_name}")
            except Exception as e:
                self.logger.error(f"Error loading section prompt {section_name}: {e}")
        
        self.logger.info(f"Loaded {len(section_prompts)} section prompts")
        return section_prompts
    
    def _initialize_gemini(self):
        """Initialize Gemini model."""
        if genai is None:
            self.logger.error("Google GenerativeAI not available")
            return None
        
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.error("GOOGLE_API_KEY environment variable not set")
            return None
        
        genai.configure(api_key=api_key)
        
        # Initialize model with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 4000,
        }
        
        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            self.logger.info(f"Initialized Gemini model: {self.model_name}")
            return model
        
        except Exception as e:
            self.logger.error(f"Error initializing Gemini model: {e}")
            return None
    
    def _initialize_token_csv(self):
        """Initialize token usage CSV file."""
        if not self.token_csv_path.exists():
            with open(self.token_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'generation_id', 'timestamp', 'model', 'persona', 
                    'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'section_type', 'generation_time_ms'
                ])
    
    def _log_token_usage(self, generation_id: str, model: str, persona: str, 
                        prompt_tokens: int, completion_tokens: int, 
                        section_type: str, generation_time_ms: float):
        """Log token usage to CSV file."""
        try:
            with open(self.token_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation_id,
                    datetime.now().isoformat(),
                    model,
                    persona,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    section_type,
                    generation_time_ms
                ])
        except Exception as e:
            self.logger.error(f"Error logging token usage: {e}")
    
    def _construct_section_prompt(self, section_type: str, user_prompt: str, 
                                 persona: str, retrieval_context: Optional[Dict] = None) -> str:
        """Construct a section-specific prompt with persona and context."""
        # Get persona information
        persona_info = self.personas.get("personas", {}).get(
            persona, 
            self.personas.get("personas", {}).get(self.personas.get("default_persona", "consultant"), {})
        )
        
        # Get section prompt template
        section_prompt = self.section_prompts.get(section_type, "")
        
        # Build context from retrieval
        context_text = ""
        if retrieval_context and retrieval_context.get("matches"):
            context_chunks = []
            for match in retrieval_context["matches"][:5]:  # Top 5 matches
                chunk_text = match.get("text", "")
                source = match.get("metadata", {}).get("source", "Unknown")
                context_chunks.append(f"Source: {source}\n{chunk_text}")
            
            context_text = "\n\n---\n\n".join(context_chunks)
        
        # Construct full prompt
        full_prompt = f"""You are an expert proposal writer. {persona_info.get('prompt_additions', '')}

{section_prompt}

## User Requirements:
{user_prompt}

## Retrieved Context:
{context_text if context_text else "No specific context provided."}

## Instructions:
- Write in {persona_info.get('writing_style', {}).get('tone', 'professional')} tone
- Focus on {', '.join(persona_info.get('writing_style', {}).get('focus_areas', ['solutions']))}
- Ensure content is relevant to the user requirements
- Use the retrieved context to inform your response
- Generate high-quality, structured content suitable for a professional proposal

Please generate the {section_type.replace('_', ' ').title()} section now:"""

        return full_prompt
    
    def _generate_section_content(self, section_type: str, user_prompt: str, 
                                 persona: str, retrieval_context: Optional[Dict] = None,
                                 generation_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate content for a specific section."""
        if self.model is None:
            raise RuntimeError("Gemini model not initialized")
        
        generation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Construct prompt
        prompt = self._construct_section_prompt(section_type, user_prompt, persona, retrieval_context)
        
        try:
            # Update generation config if parameters provided
            if generation_params:
                config = {
                    "temperature": generation_params.get("temperature", 0.7),
                    "top_p": generation_params.get("top_p", 0.9),
                    "max_output_tokens": generation_params.get("max_tokens", 4000),
                }
                
                # Create new model instance with updated config
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=config
                )
            else:
                model = self.model
            
            # Generate content
            response = model.generate_content(prompt)
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Extract content
            if response.candidates:
                markdown_content = response.candidates[0].content.parts[0].text
            else:
                raise RuntimeError("No content generated")
            
            # Convert to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'toc']
            )
            
            # Calculate word count
            word_count = len(markdown_content.split())
            
            # Extract token usage (if available)
            prompt_tokens = 0
            completion_tokens = 0
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
            
            # Log token usage
            self._log_token_usage(
                generation_id, self.model_name, persona,
                prompt_tokens, completion_tokens, section_type, generation_time_ms
            )
            
            # Extract sources referenced
            sources_referenced = []
            if retrieval_context and retrieval_context.get("matches"):
                sources_referenced = [
                    match.get("metadata", {}).get("source", "Unknown")
                    for match in retrieval_context["matches"][:5]
                ]
            
            return {
                "section_id": generation_id,
                "section_type": section_type,
                "title": section_type.replace('_', ' ').title(),
                "content": {
                    "markdown": markdown_content,
                    "html": html_content
                },
                "word_count": word_count,
                "sources_referenced": sources_referenced,
                "confidence_score": 0.8,  # Could be calculated based on response quality
                "generation_metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error generating section {section_type}: {e}")
            raise
    
    def generate(self, writer_input: WriterInput) -> WriterOutput:
        """
        Generate proposal content based on input specifications.
        
        Args:
            writer_input: Input containing prompt, persona, and context
            
        Returns:
            WriterOutput: Generated content with metadata
        """
        start_time = time.time()
        generation_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting content generation {generation_id}")
        self.logger.info(f"Persona: {writer_input.persona}")
        self.logger.info(f"Sections: {writer_input.sections_to_generate}")
        
        try:
            sections = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            # Generate each section
            for section_type in writer_input.sections_to_generate:
                self.logger.info(f"Generating section: {section_type}")
                
                section_data = self._generate_section_content(
                    section_type=section_type,
                    user_prompt=writer_input.user_prompt,
                    persona=writer_input.persona,
                    retrieval_context=writer_input.retrieval_context,
                    generation_params=writer_input.generation_params
                )
                
                sections.append(Section(**section_data))
                
                # Accumulate token usage
                metadata = section_data.get("generation_metadata", {})
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
            
            # Combine all sections into full content
            full_markdown = "\n\n".join([
                f"# {section.title}\n\n{section.content['markdown']}"
                for section in sections
            ])
            
            full_html = markdown.markdown(
                full_markdown,
                extensions=['tables', 'fenced_code', 'toc']
            )
            
            total_word_count = sum(section.word_count for section in sections)
            estimated_reading_time = max(1, total_word_count // 250)  # ~250 words per minute
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Create output structure
            output = WriterOutput(
                generation_id=generation_id,
                timestamp=datetime.now().isoformat(),
                input_context={
                    "user_prompt": writer_input.user_prompt,
                    "persona_used": writer_input.persona,
                    "retrieval_context": {
                        "retrieval_id": writer_input.retrieval_context.get("id", "") if writer_input.retrieval_context else "",
                        "total_chunks_used": len(writer_input.retrieval_context.get("matches", [])) if writer_input.retrieval_context else 0,
                        "primary_sources": list(set([
                            match.get("metadata", {}).get("source", "Unknown")
                            for match in writer_input.retrieval_context.get("matches", [])
                        ])) if writer_input.retrieval_context else []
                    }
                },
                generated_content={
                    "sections": [section.dict() for section in sections],
                    "full_content": {
                        "markdown": full_markdown,
                        "html": full_html
                    },
                    "word_count": total_word_count,
                    "estimated_reading_time": estimated_reading_time
                },
                generation_metadata={
                    "model_used": self.model_name,
                    "model_version": "2.5-flash",  # Gemini 2.5 Flash
                    "generation_time_ms": generation_time_ms,
                    "token_usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens
                    },
                    "generation_parameters": writer_input.generation_params
                }
            )
            
            self.logger.info(f"Content generation completed: {generation_id}")
            self.logger.info(f"Total word count: {total_word_count}")
            self.logger.info(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")
            
            return output
        
        except Exception as e:
            self.logger.error(f"Error in content generation: {e}")
            raise
    
    def save_result(self, output: WriterOutput, output_dir: str = "data/generated") -> str:
        """
        Save generation result to files.
        
        Args:
            output: WriterOutput to save
            output_dir: Directory to save files
            
        Returns:
            str: Path to saved JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON output
        json_file = output_path / f"writer_output_{output.generation_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output.dict(), f, indent=2, ensure_ascii=False)
        
        # Save markdown file
        md_file = output_path / f"proposal_{output.generation_id}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(output.generated_content["full_content"]["markdown"])
        
        # Save HTML file
        html_file = output_path / f"proposal_{output.generation_id}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(output.generated_content["full_content"]["html"])
        
        self.logger.info(f"Results saved to: {json_file}")
        return str(json_file)


# Example usage
def main():
    """Example usage of the Writer Agent."""
    # Initialize agent
    agent = WriterAgent()
    
    # Example input
    writer_input = WriterInput(
        user_prompt="We need a web application for customer management with user authentication, data analytics, and reporting capabilities.",
        persona="technical",
        sections_to_generate=["executive_summary", "technical_approach"]
    )
    
    try:
        # Generate content
        result = agent.generate(writer_input)
        
        # Save results
        saved_path = agent.save_result(result)
        print(f"Generated proposal saved to: {saved_path}")
        
        # Display summary
        print(f"\nGeneration Summary:")
        print(f"- Generation ID: {result.generation_id}")
        print(f"- Word Count: {result.generated_content['word_count']}")
        print(f"- Sections: {len(result.generated_content['sections'])}")
        print(f"- Total Tokens: {result.generation_metadata['token_usage']['total_tokens']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 