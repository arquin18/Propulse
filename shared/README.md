# Propulse Shared Resources

This directory contains shared resources, schemas, and templates used across the Propulse system. It ensures consistency between frontend and backend components.

## 📁 Directory Structure

```
shared/
├── mcp_schemas/     # Model Context Protocol schemas
│   ├── input/       # Input schemas
│   └── output/      # Output schemas
├── sample_rfps/     # Sample RFP documents
│   ├── tech/        # Technology sector
│   └── business/    # Business sector
└── templates/       # Proposal templates
    ├── basic/       # Basic templates
    └── advanced/    # Advanced templates
```

## 📋 MCP Schemas

The Model Context Protocol (MCP) schemas define the standardized format for agent communication:

### Input Schemas
- `prompt_schema.json`: User prompt structure
- `document_schema.json`: RFP document structure
- `context_schema.json`: Context retrieval structure

### Output Schemas
- `proposal_schema.json`: Generated proposal structure
- `verification_schema.json`: Verification results structure
- `status_schema.json`: Process status structure

## 📄 Sample RFPs

Collection of sample RFP documents for testing and development:

### Technology Sector
- Software development
- Cloud infrastructure
- Cybersecurity
- AI/ML projects

### Business Sector
- Consulting services
- Business process optimization
- Market research
- Strategic planning

## 📝 Templates

Standard templates for proposal generation:

### Basic Templates
- Executive summary
- Project scope
- Timeline and milestones
- Budget breakdown

### Advanced Templates
- Technical specifications
- Risk assessment
- Implementation strategy
- Quality assurance plan

## 🔧 Usage

1. **Using MCP Schemas**
   ```python
   from shared.mcp_schemas.input import PromptSchema
   
   # Validate input
   prompt = PromptSchema(
       text="Project proposal",
       requirements=["timeline", "budget"]
   )
   ```

2. **Loading Templates**
   ```python
   from shared.templates import load_template
   
   # Load specific template
   template = load_template("basic/executive_summary")
   ```

3. **Accessing Sample RFPs**
   ```python
   from shared.sample_rfps import get_sample
   
   # Get sample RFP
   rfp = get_sample("tech/software_development")
   ```

## 🔄 Updates

1. **Adding New Schemas**
   - Create schema file in appropriate directory
   - Update schema registry
   - Add validation tests

2. **Contributing Templates**
   - Follow template guidelines
   - Include metadata
   - Add usage examples

3. **Sample RFP Guidelines**
   - Remove sensitive information
   - Standardize format
   - Include metadata

## 📚 Documentation

- [MCP Schema Reference](docs/mcp_schemas.md)
- [Template Guide](docs/templates.md)
- [Sample RFP Guide](docs/sample_rfps.md)

## 🔒 Security

- No sensitive data in samples
- Sanitized templates
- Version controlled schemas

## 🤝 Contributing

1. Follow naming conventions
2. Update documentation
3. Add tests
4. Create pull request 