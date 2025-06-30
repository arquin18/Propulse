# Propulse Frontend

The frontend service is built with Streamlit and provides an intuitive user interface for interacting with the Propulse system. It handles document uploads, prompt input, and proposal visualization.

## 📁 Directory Structure

```
frontend/
├── assets/          # Static assets (images, styles)
├── components/      # Reusable UI components
│   ├── upload/      # File upload components
│   ├── prompt/      # Prompt input components
│   └── preview/     # Proposal preview components
├── pages/          # Application pages
│   ├── home.py     # Home page
│   ├── generate.py # Proposal generation page
│   └── history.py  # Proposal history page
└── main.py        # Application entry point
```

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   conda activate propulse
   ```

2. **Start the Application**
   ```bash
   streamlit run main.py
   ```

3. **Access the UI**
   - Open http://localhost:8501 in your browser

## 🎨 UI Components

### Document Upload
- Supports PDF and DOCX files
- Drag-and-drop interface
- File size validation
- Preview functionality

### Prompt Input
- Rich text editor
- Template suggestions
- Character count
- History tracking

### Proposal Preview
- Real-time updates
- Export options
- Version comparison
- Comment system

## 🔧 Configuration

The frontend is configured through `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = true
```

## 🎯 Features

1. **Responsive Design**
   - Mobile-friendly layout
   - Adaptive components
   - Touch support

2. **Real-time Updates**
   - WebSocket connection
   - Progress indicators
   - Status notifications

3. **Data Visualization**
   - Progress charts
   - Success metrics
   - Usage statistics

## 🛠️ Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Format Code**
   ```bash
   black .
   isort .
   ```

## 🔍 Debugging

1. **Enable Debug Mode**
   ```bash
   streamlit run main.py --logger.level=debug
   ```

2. **Browser Developer Tools**
   - Network tab for API calls
   - Console for JavaScript logs
   - Elements for component inspection

## 🎨 Styling

1. **CSS Customization**
   - Edit `assets/style.css`
   - Use Streamlit theme configuration
   - Custom component styling

2. **Component Themes**
   - Light/dark mode support
   - Custom color schemes
   - Responsive layouts

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Components](https://streamlit.io/components)
- [Streamlit Forum](https://discuss.streamlit.io/)

## 🤝 Contributing

1. Follow the style guide
2. Write unit tests
3. Update documentation
4. Create pull request

## 🐛 Known Issues

- See GitHub Issues for current problems
- Check CHANGELOG.md for recent fixes
- Report new issues with templates 