# Clinical NLP Streamlit Web Application

This directory contains a fully interactive web application built with Streamlit for clinical NLP entity extraction from neonatal discharge summaries.

## 🚀 Quick Start

### Windows Users
1. Double-click `launch_streamlit.bat`
2. The application will automatically install dependencies and launch in your browser

### Mac/Linux Users
1. Make the script executable: `chmod +x launch_streamlit.sh`
2. Run: `./launch_streamlit.sh`
3. The application will automatically install dependencies and launch in your browser

### Manual Installation
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Launch the application
streamlit run streamlit_app.py
```

## 🎯 Features

### 1. Single Text Analysis
- **Manual Text Input**: Enter discharge summaries directly
- **Sample Text Selection**: Choose from pre-loaded examples
- **File Upload**: Upload text files for analysis
- **Ground Truth Evaluation**: Optional evaluation with known correct answers
- **Real-time Processing**: Instant entity extraction and visualization

### 2. Interactive Output Formats
- **Entity Cards**: Visual display of extracted entities with success/failure indicators
- **JSON Output**: Structured data format for API integration
- **Markdown Tables**: Publication-ready table format
- **Interactive Visualizations**: Charts and graphs showing extraction results

### 3. Batch Processing
- **Multiple Text Processing**: Handle multiple discharge summaries at once
- **CSV File Upload**: Batch process from CSV files
- **Progress Tracking**: Real-time progress bars during processing
- **Aggregate Statistics**: Overall performance metrics across all texts

### 4. Advanced Evaluation
- **Precision, Recall, F1-Score**: Standard NLP evaluation metrics
- **Per-Entity Analysis**: Individual performance for each entity type
- **Visual Metrics**: Interactive charts showing performance data
- **Comparison Analysis**: Compare extracted vs. ground truth entities

### 5. Sample Data Explorer
- **Pre-loaded Examples**: 5+ sample neonatal discharge summaries
- **Ground Truth Annotations**: Complete entity annotations for each sample
- **Test Extraction**: Verify system performance on known examples

### 6. System Information
- **Pattern Viewer**: Explore the regex patterns used for extraction
- **Usage Examples**: Sample texts and expected outputs
- **Performance Guidelines**: Tips for better extraction results
- **Processing History**: Track your analysis session

## 🎨 User Interface Features

### Modern Web Design
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Tabs**: Organized content in easy-to-navigate tabs
- **Color-coded Results**: Green for found entities, red for missing
- **Progress Indicators**: Visual feedback during processing
- **Custom Styling**: Professional medical application appearance

### User Experience
- **Guided Workflow**: Step-by-step process from input to results
- **Multiple Input Methods**: Text entry, file upload, or sample selection
- **Real-time Feedback**: Instant validation and error messages
- **Export Options**: Download results in various formats
- **Session Memory**: Maintains processing history during session

## 📊 Supported Entities

The system extracts 8 key clinical entities:

1. **Patient ID** (P_ID): Medical record numbers, patient identifiers
2. **Gestational Age**: Birth age in weeks (e.g., "34 weeks", "32+3 weeks")
3. **Sex**: Gender classification (Male/Female)
4. **Birth Weight**: Weight at birth with units (e.g., "2100 grams", "1.8 kg")
5. **Diagnosis**: Medical conditions and diagnoses
6. **Treatment_Respiratory**: Respiratory support and interventions
7. **Treatment_Medication**: Medications and pharmaceutical treatments
8. **Outcome**: Discharge status and patient outcomes

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Backend**: Python-based NLP processing
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation
- **Pattern Matching**: Regex-based clinical entity recognition

### Performance
- **Real-time Processing**: Instant results for single texts
- **Batch Processing**: Efficient handling of multiple documents
- **Memory Efficient**: Optimized for clinical text processing
- **Scalable**: Handles varying text lengths and complexity

## 📈 Evaluation Metrics

### Precision
- **Definition**: Correct extractions / Total extractions
- **Interpretation**: How many extracted entities were actually correct
- **Range**: 0.0 to 1.0 (higher is better)

### Recall  
- **Definition**: Correct extractions / Total actual entities
- **Interpretation**: How many actual entities were successfully found
- **Range**: 0.0 to 1.0 (higher is better)

### F1-Score
- **Definition**: Harmonic mean of Precision and Recall
- **Interpretation**: Balanced measure of extraction performance
- **Range**: 0.0 to 1.0 (higher is better)

## 🎯 Usage Scenarios

### Clinical Research
- **Retrospective Studies**: Extract entities from historical discharge summaries
- **Data Analysis**: Generate structured data from unstructured clinical text
- **Quality Assessment**: Evaluate documentation completeness

### Healthcare Quality Improvement
- **Documentation Review**: Identify missing critical information
- **Standardization**: Promote consistent discharge summary formatting
- **Training**: Educational tool for medical documentation

### Academic Use
- **NLP Research**: Benchmark clinical entity extraction systems
- **Medical Informatics**: Study clinical text processing techniques
- **Student Projects**: Learn clinical NLP concepts interactively

## 🔍 Example Workflows

### Workflow 1: Single Text Analysis
1. Select "Single Text Analysis" mode
2. Choose "Enter text manually"
3. Paste a discharge summary
4. Click "Extract Entities"
5. Review results in multiple formats
6. Optionally provide ground truth for evaluation

### Workflow 2: Batch Processing
1. Select "Batch Processing" mode
2. Choose "Use all sample data" or upload CSV
3. Click "Process Batch"
4. Review aggregate metrics and individual results
5. Export results for further analysis

### Workflow 3: Sample Exploration
1. Select "Sample Data Explorer" mode
2. Browse through pre-loaded examples
3. View ground truth annotations
4. Test extraction on known examples
5. Compare system performance

## 🎨 Customization Options

### Styling
- **Custom CSS**: Modify appearance and branding
- **Color Themes**: Adjust color schemes for entities
- **Layout Options**: Configure sidebar and main content areas

### Functionality
- **Entity Types**: Add or modify clinical entities
- **Pattern Rules**: Customize extraction patterns
- **Evaluation Metrics**: Add custom performance measures

## 🚨 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Port Conflicts**: Change port in launch scripts if needed
- **File Permissions**: Make scripts executable on Mac/Linux
- **Browser Issues**: Clear cache or try different browser

### Getting Help
- Check console output for detailed error messages
- Verify all required files are in the same directory
- Ensure Python 3.8+ is installed and accessible
- Review the system information tab for guidance

## 📄 File Structure

```
model/
├── streamlit_app.py              # Main Streamlit application
├── neonatal_ner.py              # Core NLP system
├── entity_patterns.py           # Pattern definitions
├── evaluation.py                # Evaluation framework
├── requirements_streamlit.txt   # Web app dependencies
├── launch_streamlit.bat         # Windows launcher
├── launch_streamlit.sh          # Mac/Linux launcher
├── data/                        # Sample data
│   └── sample_annotations.json
└── notebooks/                   # Jupyter notebooks
    └── neonatal_ner_demo.ipynb
```

## 🎉 Getting Started

1. **Launch the Application**
   - Windows: Run `launch_streamlit.bat`
   - Mac/Linux: Run `./launch_streamlit.sh`

2. **Try the Demo**
   - Start with "Sample Data Explorer" to see examples
   - Move to "Single Text Analysis" to test your own text
   - Explore "Batch Processing" for multiple documents

3. **Explore Features**
   - Test different input methods
   - Try various visualization options
   - Experiment with ground truth evaluation

4. **Learn More**
   - Visit "System Information" for detailed help
   - Check processing history for session tracking
   - Review pattern information for technical details

The application provides a comprehensive, user-friendly interface for clinical NLP entity extraction with professional-grade evaluation and visualization capabilities.
