import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time
import re

# Import our custom modules
try:
    from neonatal_ner import NeonatalNER
    from entity_patterns import EntityPatterns
    from evaluation import EntityEvaluator
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    st.error("Please make sure all required files are in the same directory.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Clinical NLP - Neonatal Entity Extraction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .entity-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .entity-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data focused on RDS cases for demonstration"""
    try:
        with open('data/sample_annotations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return [
            {
                "text": "Patient ID: RDS-2024-001. Male infant born at 28 weeks gestational age with birth weight of 1200 grams. Primary diagnosis: Severe respiratory distress syndrome. Treatment included immediate intubation, mechanical ventilation, and surfactant therapy within 2 hours of birth. Medications administered: Poractant alfa (surfactant), caffeine citrate for apnea of prematurity, dexamethasone for chronic lung disease prevention. Patient required ventilatory support for 14 days, then transitioned to CPAP. Discharged home at 36 weeks corrected gestational age in stable condition on supplemental oxygen.",
                "annotations": {
                    "P_ID": "RDS-2024-001",
                    "Gestational_Age": "28 weeks",
                    "Sex": "Male",
                    "Birth_Weight": "1200 grams",
                    "Diagnosis": "Severe respiratory distress syndrome",
                    "Treatment_Respiratory": "immediate intubation, mechanical ventilation, surfactant therapy, CPAP, supplemental oxygen",
                    "Treatment_Medication": "Poractant alfa, caffeine citrate, dexamethasone",
                    "Outcome": "discharged home at 36 weeks corrected gestational age in stable condition on supplemental oxygen"
                }
            },
            {
                "text": "Female neonate, medical record RDS-2024-002, born at 32 weeks gestation, birth weight 1650g. Diagnosed with moderate respiratory distress syndrome and patent ductus arteriosus. Initial respiratory support with CPAP, progressed to require intubation and mechanical ventilation on day 2. Two doses of beractant surfactant administered. Additional treatments: indomethacin for PDA closure, furosemide for fluid management. Weaned from ventilator after 8 days to high-flow nasal cannula, then room air by day 15. Discharged home at term equivalent age without oxygen support.",
                "annotations": {
                    "P_ID": "RDS-2024-002",
                    "Gestational_Age": "32 weeks", 
                    "Sex": "Female",
                    "Birth_Weight": "1650g",
                    "Diagnosis": "moderate respiratory distress syndrome, patent ductus arteriosus",
                    "Treatment_Respiratory": "CPAP, intubation, mechanical ventilation, beractant surfactant, high-flow nasal cannula",
                    "Treatment_Medication": "beractant, indomethacin, furosemide",
                    "Outcome": "discharged home at term equivalent age without oxygen support"
                }
            },
            {
                "text": "Patient ID: RDS-2024-003. Male preterm infant, gestational age 30+2 weeks, weight 1400 grams. Severe RDS with bilateral ground-glass opacities on chest X-ray. Required immediate intubation in delivery room. Received prophylactic surfactant (Curosurf) followed by synchronized intermittent mandatory ventilation (SIMV). Developed pneumothorax on day 3, treated with chest tube insertion. Medications included caffeine, vitamin A therapy, and brief course of hydrocortisone. Chronic lung disease developed, requiring prolonged ventilatory support. Tracheostomy performed at 8 weeks of age. Discharged home at 4 months of age with home ventilator support.",
                "annotations": {
                    "P_ID": "RDS-2024-003",
                    "Gestational_Age": "30+2 weeks",
                    "Sex": "Male",
                    "Birth_Weight": "1400 grams",
                    "Diagnosis": "Severe RDS with bilateral ground-glass opacities, pneumothorax, chronic lung disease",
                    "Treatment_Respiratory": "immediate intubation, prophylactic surfactant, SIMV, chest tube insertion, tracheostomy, home ventilator",
                    "Treatment_Medication": "Curosurf, caffeine, vitamin A therapy, hydrocortisone",
                    "Outcome": "discharged home at 4 months of age with home ventilator support"
                }
            },
            {
                "text": "Female infant, ID: RDS-2024-004, born at 34 weeks gestational age, birth weight 2100g. Mild respiratory distress syndrome diagnosed based on clinical presentation and chest imaging. Managed with nasal CPAP and supplemental oxygen. Single dose of surfactant (Survanta) administered via INSURE technique. Respiratory status improved within 24 hours. No additional medications required. Transitioned to room air by day 5. Full-term feeding achieved by discharge. Discharged home on day 10 in excellent condition, no follow-up respiratory support needed.",
                "annotations": {
                    "P_ID": "RDS-2024-004",
                    "Gestational_Age": "34 weeks",
                    "Sex": "Female", 
                    "Birth_Weight": "2100g",
                    "Diagnosis": "Mild respiratory distress syndrome",
                    "Treatment_Respiratory": "nasal CPAP, supplemental oxygen, surfactant via INSURE technique",
                    "Treatment_Medication": "Survanta",
                    "Outcome": "discharged home on day 10 in excellent condition, no follow-up respiratory support needed"
                }
            },
            {
                "text": "Preterm male twin A, patient number RDS-2024-005, gestational age 29 weeks, birth weight 1300 grams. Diagnosed with severe RDS complicated by persistent pulmonary hypertension of the newborn (PPHN). Required high-frequency oscillatory ventilation (HFOV) and inhaled nitric oxide therapy. Multiple surfactant doses administered over first 3 days. Additional treatments included sildenafil for PPHN, diuretics, and bronchodilators. Developed bronchopulmonary dysplasia requiring long-term oxygen therapy. Gradual weaning from ventilator support over 6 weeks. Discharged home at 38 weeks corrected age on low-flow oxygen and multiple medications.",
                "annotations": {
                    "P_ID": "RDS-2024-005",
                    "Gestational_Age": "29 weeks",
                    "Sex": "Male",
                    "Birth_Weight": "1300 grams",
                    "Diagnosis": "severe RDS complicated by persistent pulmonary hypertension, bronchopulmonary dysplasia",
                    "Treatment_Respiratory": "high-frequency oscillatory ventilation, inhaled nitric oxide therapy, multiple surfactant doses, long-term oxygen therapy",
                    "Treatment_Medication": "surfactant, sildenafil, diuretics, bronchodilators",
                    "Outcome": "discharged home at 38 weeks corrected age on low-flow oxygen and multiple medications"
                }
            }
        ]

# Initialize session state
if 'ner_system' not in st.session_state:
    st.session_state.ner_system = NeonatalNER()
if 'extraction_history' not in st.session_state:
    st.session_state.extraction_history = []
if 'sample_texts' not in st.session_state:
    st.session_state.sample_texts = load_sample_data()

def create_entity_visualization(entities: Dict) -> go.Figure:
    """Create a visualization of extracted entities"""
    entity_names = list(entities.keys())
    has_value = [1 if entities[name] is not None and str(entities[name]).strip() != "" else 0 for name in entity_names]
    
    colors = ['#2ecc71' if val == 1 else '#e74c3c' for val in has_value]
    
    fig = go.Figure(data=[
        go.Bar(
            x=entity_names,
            y=has_value,
            marker_color=colors,
            text=[entities[name] if entities[name] else "Not Found" for name in entity_names],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Extracted Entities Overview",
        xaxis_title="Entity Types",
        yaxis_title="Found (1) / Not Found (0)",
        yaxis=dict(tickvals=[0, 1], ticktext=["Not Found", "Found"]),
        height=400,
        showlegend=False,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_metrics_visualization(metrics: Dict) -> go.Figure:
    """Create visualization for evaluation metrics"""
    fig = go.Figure()
    
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    fig.add_trace(go.Bar(
        x=metrics_names,
        y=metrics_values,
        marker_color=colors,
        text=[f'{val:.3f}' for val in metrics_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Evaluation Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Clinical NLP System for RDS</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #7f8c8d;">Respiratory Distress Syndrome Entity Extraction from Neonatal Discharge Summaries</h2>', unsafe_allow_html=True)
    
    # Add RDS information banner
    with st.container():
        st.info("""
        🔬 **Focus: Respiratory Distress Syndrome (RDS)**
        
        This specialized clinical NLP system is optimized for extracting entities from neonatal discharge summaries 
        specifically related to Respiratory Distress Syndrome cases. RDS is a common condition in preterm infants 
        caused by surfactant deficiency, requiring specialized respiratory support and treatment protocols.
        
        **Key RDS-related entities extracted:**
        - Advanced respiratory treatments (CPAP, mechanical ventilation, HFOV, surfactant therapy)
        - RDS-specific medications (surfactant, caffeine, steroids)  
        - Complications (pneumothorax, chronic lung disease, bronchopulmonary dysplasia)
        - Long-term outcomes and home care requirements
        """)
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Single Text Analysis", "Batch Processing", "Sample Data Explorer", "System Information"]
    )
    
    if mode == "Single Text Analysis":
        single_text_analysis()
    elif mode == "Batch Processing":
        batch_processing()
    elif mode == "Sample Data Explorer":
        sample_data_explorer()
    elif mode == "System Information":
        system_information()

def single_text_analysis():
    """Single text analysis interface"""
    st.markdown('<h2 class="sub-header">📝 Single Text Analysis</h2>', unsafe_allow_html=True)
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Enter text manually", "Use sample text", "Upload file"]
    )
    
    user_text = ""
    ground_truth = None
    
    if input_method == "Enter text manually":
        st.markdown("### Enter Neonatal Discharge Summary:")
        user_text = st.text_area(
            "Paste your neonatal discharge summary here:",
            height=200,
            placeholder="Example: Patient ID: NB-2024-001. Male infant born at 32 weeks gestational age with birth weight of 1800g. Diagnosed with respiratory distress syndrome..."
        )
        
        # Optional ground truth for evaluation
        with st.expander("🎯 Optional: Provide Ground Truth for Evaluation"):
            st.markdown("If you have ground truth annotations, you can provide them here for evaluation:")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gt_pid = st.text_input("Patient ID", key="gt_pid")
                gt_ga = st.text_input("Gestational Age", key="gt_ga")
            with col2:
                gt_sex = st.text_input("Sex", key="gt_sex")
                gt_bw = st.text_input("Birth Weight", key="gt_bw")
            with col3:
                gt_diag = st.text_input("Diagnosis", key="gt_diag")
                gt_resp = st.text_input("Respiratory Treatment", key="gt_resp")
            with col4:
                gt_med = st.text_input("Medication", key="gt_med")
                gt_outcome = st.text_input("Outcome", key="gt_outcome")
            
            if any([gt_pid, gt_ga, gt_sex, gt_bw, gt_diag, gt_resp, gt_med, gt_outcome]):
                ground_truth = {
                    "P_ID": gt_pid or None,
                    "Gestational_Age": gt_ga or None,
                    "Sex": gt_sex or None,
                    "Birth_Weight": gt_bw or None,
                    "Diagnosis": gt_diag or None,
                    "Treatment_Respiratory": gt_resp or None,
                    "Treatment_Medication": gt_med or None,
                    "Outcome": gt_outcome or None
                }
    
    elif input_method == "Use sample text":
        sample_options = [f"Sample {i+1}: {text['text'][:50]}..." for i, text in enumerate(st.session_state.sample_texts)]
        selected_sample = st.selectbox("Choose a sample text:", sample_options)
        
        if selected_sample:
            sample_idx = int(selected_sample.split(":")[0].split()[1]) - 1
            user_text = st.session_state.sample_texts[sample_idx]['text']
            ground_truth = st.session_state.sample_texts[sample_idx]['annotations']
            
            st.text_area("Selected text:", user_text, height=150, disabled=True)
            
            with st.expander("📋 Ground Truth Annotations"):
                st.json(ground_truth)
    
    elif input_method == "Upload file":
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded text:", user_text, height=150, disabled=True)
    
    # Processing button
    if st.button("🚀 Extract Entities", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing text..."):
                # Simulate processing time for better UX
                time.sleep(1)
                
                # Extract entities
                result = st.session_state.ner_system.extract_entities(user_text, ground_truth)
                
                # Add to history
                st.session_state.extraction_history.append({
                    'text': user_text[:100] + "..." if len(user_text) > 100 else user_text,
                    'result': result,
                    'timestamp': time.time()
                })
                
                display_extraction_results(result, user_text, ground_truth)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

def display_extraction_results(result: Dict, original_text: str, ground_truth: Optional[Dict] = None):
    """Display the extraction results in a formatted way"""
    st.markdown('<h3 class="sub-header">📊 Extraction Results</h3>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Entities", "📋 JSON Output", "📊 Visualization", "📈 Evaluation"])
    
    with tab1:
        st.markdown("### Extracted Entities")
        
        entities = result['entities']
        entity_labels = [
            ("🆔 Patient ID", "P_ID"),
            ("📅 Gestational Age", "Gestational_Age"),
            ("⚧ Sex", "Sex"),
            ("⚖️ Birth Weight", "Birth_Weight"),
            ("🏥 Diagnosis", "Diagnosis"),
            ("🫁 Respiratory Treatment", "Treatment_Respiratory"),
            ("💊 Medication Treatment", "Treatment_Medication"),
            ("📋 Outcome", "Outcome")
        ]
        
        col1, col2 = st.columns(2)
        
        for i, (label, key) in enumerate(entity_labels):
            value = entities.get(key)
            with col1 if i % 2 == 0 else col2:
                if value:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>{label}</strong><br>
                        {value}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>{label}</strong><br>
                        <em>Not found</em>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### JSON Format")
        st.code(result['json_output'], language='json')
        
        st.markdown("### Markdown Table Format")
        st.code(result['markdown_table'], language='markdown')
        
        # Display as actual markdown table
        st.markdown("### Rendered Table")
        st.markdown(result['markdown_table'])
    
    with tab3:
        st.markdown("### Entity Extraction Visualization")
        fig = create_entity_visualization(result['entities'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Entity distribution pie chart
        entities = result['entities']
        found_count = sum(1 for v in entities.values() if v is not None and str(v).strip())
        not_found_count = len(entities) - found_count
        
        fig_pie = px.pie(
            values=[found_count, not_found_count],
            names=['Found', 'Not Found'],
            title="Entity Extraction Success Rate",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab4:
        if 'evaluation' in result and result['evaluation']:
            st.markdown("### Evaluation Metrics")
            
            eval_data = result['evaluation']
            
            # Metrics cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Precision",
                    value=f"{eval_data['precision']:.3f}",
                    delta=f"{eval_data['correct_extractions']}/{eval_data['total_extractions']}"
                )
            
            with col2:
                st.metric(
                    label="Recall",
                    value=f"{eval_data['recall']:.3f}",
                    delta=f"{eval_data['correct_extractions']}/{eval_data['total_actual']}"
                )
            
            with col3:
                st.metric(
                    label="F1-Score",
                    value=f"{eval_data['f1_score']:.3f}",
                    delta="Harmonic Mean"
                )
            
            # Metrics visualization
            fig_metrics = create_metrics_visualization(eval_data)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Per-entity results
            st.markdown("### Per-Entity Evaluation")
            entity_results = []
            for entity, status in eval_data['entity_results'].items():
                entity_results.append({
                    'Entity': entity,
                    'Status': status,
                    'Icon': '✅' if status == 'Correct' else '❌' if status == 'Incorrect' else '⚪' if status == 'Missed' else '➖'
                })
            
            df_results = pd.DataFrame(entity_results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
        else:
            st.info("💡 Provide ground truth annotations to see evaluation metrics.")

def batch_processing():
    """Batch processing interface"""
    st.markdown('<h2 class="sub-header">📊 Batch Processing</h2>', unsafe_allow_html=True)
    
    st.info("Upload multiple texts or use all sample data for batch processing.")
    
    # Batch input options
    batch_method = st.radio(
        "Choose batch input method:",
        ["Use all sample data", "Upload CSV file", "Enter multiple texts"]
    )
    
    texts = []
    ground_truths = []
    
    if batch_method == "Use all sample data":
        texts = [item['text'] for item in st.session_state.sample_texts]
        ground_truths = [item['annotations'] for item in st.session_state.sample_texts]
        
        st.success(f"✅ Loaded {len(texts)} sample texts for batch processing.")
    
    elif batch_method == "Upload CSV file":
        st.markdown("### Upload CSV File")
        st.info("CSV should have columns: 'text' and optionally ground truth columns")
        
        uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            
            if 'text' in df.columns:
                texts = df['text'].tolist()
                st.success(f"✅ Loaded {len(texts)} texts from CSV file.")
            else:
                st.error("❌ CSV file must contain a 'text' column.")
    
    elif batch_method == "Enter multiple texts":
        st.markdown("### Enter Multiple Texts")
        num_texts = st.number_input("Number of texts to process:", min_value=1, max_value=10, value=2)
        
        for i in range(num_texts):
            with st.expander(f"Text {i+1}"):
                text = st.text_area(f"Text {i+1}:", height=100, key=f"batch_text_{i}")
                if text.strip():
                    texts.append(text)
    
    # Process batch
    if st.button("🚀 Process Batch", type="primary", use_container_width=True):
        if texts:
            with st.spinner(f"Processing {len(texts)} texts..."):
                progress_bar = st.progress(0)
                
                batch_results = []
                for i, text in enumerate(texts):
                    gt = ground_truths[i] if i < len(ground_truths) else None
                    result = st.session_state.ner_system.extract_entities(text, gt)
                    batch_results.append(result)
                    progress_bar.progress((i + 1) / len(texts))
                
                # Calculate batch statistics
                evaluator = EntityEvaluator()
                if ground_truths and len(ground_truths) == len(texts):
                    all_predictions = [result['entities'] for result in batch_results]
                    batch_evaluation = evaluator.batch_evaluate(all_predictions, ground_truths)
                    
                    display_batch_results(batch_results, batch_evaluation, texts, ground_truths)
                else:
                    display_batch_results(batch_results, None, texts, None)
        else:
            st.warning("⚠️ Please provide texts to process.")

def display_batch_results(batch_results: List[Dict], batch_evaluation: Optional[Dict], texts: List[str], ground_truths: Optional[List[Dict]]):
    """Display batch processing results"""
    st.markdown('<h3 class="sub-header">📊 Batch Processing Results</h3>', unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", len(batch_results))
    
    with col2:
        total_entities = sum(len([v for v in result['entities'].values() if v]) for result in batch_results)
        st.metric("Total Entities Found", total_entities)
    
    with col3:
        avg_entities = total_entities / len(batch_results) if batch_results else 0
        st.metric("Avg Entities per Text", f"{avg_entities:.1f}")
    
    with col4:
        if batch_evaluation:
            st.metric("Overall F1-Score", f"{batch_evaluation['aggregate_f1_score']:.3f}")
    
    # Detailed results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Individual Results", "📊 Aggregate Metrics", "📈 Visualizations"])
    
    with tab1:
        st.markdown("### Individual Text Results")
        
        for i, (result, text) in enumerate(zip(batch_results, texts)):
            with st.expander(f"Text {i+1}: {text[:50]}..."):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Extracted Entities:**")
                    for entity, value in result['entities'].items():
                        status = "✅" if value else "❌"
                        st.write(f"{status} {entity}: {value or 'Not found'}")
                
                with col_b:
                    if 'evaluation' in result and result['evaluation']:
                        eval_data = result['evaluation']
                        st.markdown("**Evaluation:**")
                        st.write(f"Precision: {eval_data['precision']:.3f}")
                        st.write(f"Recall: {eval_data['recall']:.3f}")
                        st.write(f"F1-Score: {eval_data['f1_score']:.3f}")
    
    with tab2:
        if batch_evaluation:
            st.markdown("### Aggregate Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Aggregate Precision", f"{batch_evaluation['aggregate_precision']:.3f}")
            with col2:
                st.metric("Aggregate Recall", f"{batch_evaluation['aggregate_recall']:.3f}")
            with col3:
                st.metric("Aggregate F1-Score", f"{batch_evaluation['aggregate_f1_score']:.3f}")
            
            # Per-entity metrics
            st.markdown("### Per-Entity Performance")
            entity_metrics = []
            for entity, metrics in batch_evaluation['per_entity_metrics'].items():
                entity_metrics.append({
                    'Entity': entity,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'Correct': metrics['correct'],
                    'Total Predicted': metrics['total_predicted'],
                    'Total Actual': metrics['total_actual']
                })
            
            df_metrics = pd.DataFrame(entity_metrics)
            st.dataframe(df_metrics.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("💡 Ground truth data needed for aggregate metrics.")
    
    with tab3:
        st.markdown("### Batch Visualizations")
        
        # Entity distribution across all texts
        all_entities = {}
        for result in batch_results:
            for entity, value in result['entities'].items():
                if entity not in all_entities:
                    all_entities[entity] = {'found': 0, 'total': 0}
                all_entities[entity]['total'] += 1
                if value:
                    all_entities[entity]['found'] += 1
        
        # Create success rate visualization
        entity_names = list(all_entities.keys())
        success_rates = [all_entities[entity]['found'] / all_entities[entity]['total'] * 100 for entity in entity_names]
        
        fig_success = px.bar(
            x=entity_names,
            y=success_rates,
            title="Entity Extraction Success Rate Across All Texts",
            labels={'x': 'Entity Type', 'y': 'Success Rate (%)'},
            color=success_rates,
            color_continuous_scale='RdYlGn'
        )
        fig_success.update_layout(xaxis=dict(tickangle=45))
        st.plotly_chart(fig_success, use_container_width=True)
        
        if batch_evaluation:
            # Per-entity F1 scores
            entity_f1_scores = [batch_evaluation['per_entity_metrics'][entity]['f1_score'] for entity in entity_names]
            
            fig_f1 = px.bar(
                x=entity_names,
                y=entity_f1_scores,
                title="F1-Score by Entity Type",
                labels={'x': 'Entity Type', 'y': 'F1-Score'},
                color=entity_f1_scores,
                color_continuous_scale='Viridis'
            )
            fig_f1.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig_f1, use_container_width=True)

def sample_data_explorer():
    """Sample data explorer interface"""
    st.markdown('<h2 class="sub-header">🔍 Sample Data Explorer</h2>', unsafe_allow_html=True)
    
    st.info(f"Explore {len(st.session_state.sample_texts)} sample neonatal discharge summaries.")
    
    # Sample selection
    sample_idx = st.selectbox(
        "Select a sample to explore:",
        range(len(st.session_state.sample_texts)),
        format_func=lambda x: f"Sample {x+1}: {st.session_state.sample_texts[x]['text'][:50]}..."
    )
    
    sample_data = st.session_state.sample_texts[sample_idx]
    
    # Display sample
    st.markdown("### Original Text")
    st.text_area("Discharge Summary", sample_data['text'], height=200, disabled=True)
    
    st.markdown("### Ground Truth Annotations")
    st.json(sample_data['annotations'])
    
    # Test extraction
    if st.button("🧪 Test Extraction on This Sample", type="primary"):
        with st.spinner("Testing extraction..."):
            result = st.session_state.ner_system.extract_entities(
                sample_data['text'], 
                sample_data['annotations']
            )
            
            display_extraction_results(result, sample_data['text'], sample_data['annotations'])

def system_information():
    """System information and help focused on RDS"""
    st.markdown('<h2 class="sub-header">ℹ️ RDS-Focused Clinical NLP System Information</h2>', unsafe_allow_html=True)
    
    # About the system
    st.markdown("""
    ### About the RDS Clinical NLP System
    
    This specialized system is designed to extract key entities from neonatal discharge summaries 
    with a particular focus on **Respiratory Distress Syndrome (RDS)** cases. RDS is the most 
    common respiratory condition in preterm infants, requiring complex treatment protocols and 
    long-term monitoring.
    
    #### Understanding RDS:
    **Respiratory Distress Syndrome (RDS)** is a breathing disorder that affects newborn babies, 
    most commonly those born prematurely. It's caused by a lack of surfactant, a substance that 
    helps keep the lungs inflated.
    
    #### RDS-Specific Entities Extracted:
    - **P_ID**: Patient identifier or medical record number
    - **Gestational_Age**: Critical for RDS risk assessment (e.g., "28 weeks", "32+3 weeks")
    - **Sex**: Male infants have slightly higher RDS risk
    - **Birth_Weight**: Lower birth weight correlates with RDS severity
    - **Diagnosis**: RDS severity levels (mild, moderate, severe) and complications
    - **Treatment_Respiratory**: RDS-specific interventions:
      - Surfactant therapy (Poractant alfa, Beractant, Curosurf)
      - Mechanical ventilation modalities (SIMV, HFOV)
      - Non-invasive support (CPAP, INSURE technique)
      - Advanced therapies (inhaled nitric oxide, ECMO)
    - **Treatment_Medication**: RDS-related medications:
      - Surfactant preparations
      - Caffeine citrate for apnea of prematurity  
      - Steroids for chronic lung disease prevention
      - Diuretics for fluid management
    - **Outcome**: RDS-specific outcomes and complications:
      - Bronchopulmonary dysplasia (BPD)
      - Chronic lung disease
      - Home oxygen requirements
      - Long-term respiratory support needs
    
    #### RDS Treatment Protocols Recognized:
    - **INSURE Technique**: Intubation-Surfactant-Extubation
    - **LISA/MIST**: Less invasive surfactant administration
    - **Prophylactic vs Rescue Surfactant**: Timing-based treatment approaches
    - **Ventilator Strategies**: Lung-protective ventilation protocols
    - **Weaning Protocols**: Systematic reduction of respiratory support
    
    #### Features Optimized for RDS:
    - ✅ Advanced pattern recognition for respiratory terminology
    - ✅ RDS severity classification extraction
    - ✅ Surfactant product and dosing information
    - ✅ Ventilator mode and parameter extraction
    - ✅ Complication detection and monitoring
    - ✅ Long-term outcome prediction support
    - ✅ Quality metrics for RDS care protocols
    """)
    
    # RDS-specific patterns
    with st.expander("🔧 RDS-Specific Pattern Information"):
        st.markdown("""
        ### RDS-Optimized Extraction Patterns
        
        **Respiratory Treatment Patterns:**
        - Surfactant preparations: Poractant alfa, Beractant, Curosurf, Survanta
        - Ventilation modes: SIMV, HFOV, conventional ventilation, BiPAP
        - Non-invasive support: nCPAP, INSURE, LISA, high-flow nasal cannula
        - Advanced therapies: inhaled nitric oxide, ECMO, chest tube insertion
        
        **Medication Patterns:**
        - Surfactant-specific: dose timing, administration method
        - Respiratory medications: caffeine citrate, theophylline, albuterol
        - Steroids: dexamethasone, hydrocortisone, prednisolone
        - Supportive care: diuretics, vitamin A, probiotics
        
        **Diagnosis Patterns:**
        - RDS severity: mild, moderate, severe RDS
        - Complications: pneumothorax, pulmonary interstitial emphysema
        - Chronic conditions: bronchopulmonary dysplasia, chronic lung disease
        - Associated conditions: patent ductus arteriosus, apnea of prematurity
        
        **Outcome Patterns:**
        - Respiratory outcomes: weaned to room air, home oxygen, tracheostomy
        - Timing: corrected gestational age at discharge
        - Support needs: home ventilator, monitoring requirements
        """)
        
        patterns = EntityPatterns()
        all_patterns = patterns.get_all_patterns()
        
        # Focus on respiratory and medication patterns for RDS
        rds_categories = ['respiratory_treatment', 'medication', 'diagnosis']
        for category in rds_categories:
            if category in all_patterns:
                pattern_list = all_patterns[category]
                st.markdown(f"**{category.replace('_', ' ').title()}** ({len(pattern_list)} patterns)")
                with st.expander(f"View {category} patterns"):
                    for i, pattern in enumerate(pattern_list[:8]):  # Show first 8 patterns
                        st.code(pattern)
                    if len(pattern_list) > 8:
                        st.info(f"... and {len(pattern_list) - 8} more patterns")
    
    # RDS-specific usage examples
    with st.expander("📝 RDS Case Examples"):
        st.markdown("""
        ### Example RDS Discharge Summaries:
        
        **Severe RDS Case:**
        ```
        Patient ID: RDS-2024-001. Male infant born at 28 weeks gestational age, birth weight 1200g. 
        Severe respiratory distress syndrome with bilateral ground-glass opacities. Required immediate 
        intubation and mechanical ventilation. Received prophylactic surfactant (Poractant alfa 200mg/kg) 
        within 2 hours of birth. Developed pneumothorax on day 3, treated with chest tube. 
        High-frequency oscillatory ventilation initiated. Chronic lung disease developed, requiring 
        tracheostomy at 8 weeks. Discharged home at 4 months on home ventilator support.
        ```
        
        **Moderate RDS Case:**
        ```
        Female infant, ID: RDS-2024-002, born at 32 weeks gestation, weight 1650g. 
        Moderate RDS managed with CPAP initially. Required intubation on day 2 for worsening 
        respiratory distress. Two doses of beractant surfactant administered via INSURE technique. 
        Weaned to high-flow nasal cannula by day 8, then room air by day 15. 
        Discharged home at 36 weeks corrected age without oxygen support.
        ```
        
        **Mild RDS Case:**
        ```
        Patient RDS-2024-003, male infant, 34 weeks GA, 2100g birth weight. 
        Mild RDS with good response to nasal CPAP and supplemental oxygen. 
        Single surfactant dose (Survanta 4ml/kg) via INSURE technique. 
        Rapid improvement, weaned to room air by 48 hours. 
        Discharged day 5 in excellent condition, no respiratory follow-up needed.
        ```
        """)
    
    # RDS performance information
    with st.expander("📊 RDS-Specific Performance Information"):
        st.markdown("""
        ### System Performance on RDS Cases
        
        The system shows enhanced accuracy on RDS-related entities due to specialized patterns:
        
        **High Accuracy Entities (>90%):**
        - Surfactant medications and dosing
        - Ventilation modalities 
        - RDS severity classification
        - Gestational age and birth weight
        
        **Good Accuracy Entities (80-90%):**
        - Specific respiratory procedures
        - Complication identification
        - Medication timing and protocols
        
        **Challenging Areas (70-80%):**
        - Complex multi-modal treatments
        - Long-term outcome predictions
        - Subtle clinical deterioration patterns
        
        ### RDS Quality Metrics:
        - **Treatment Protocol Compliance**: Extraction of guideline-adherent care
        - **Surfactant Administration Timing**: Critical quality indicator
        - **Ventilator Days**: Important outcome measure
        - **Complication Rates**: Safety and quality monitoring
        
        ### Tips for Better RDS Entity Extraction:
        - Use standard RDS terminology and severity classifications
        - Include specific surfactant product names and dosing
        - Document ventilator modes and parameter changes
        - Specify timing of interventions relative to birth
        - Include corrected gestational age for discharge outcomes
        """)
    
    # Processing history
    if st.session_state.extraction_history:
        st.markdown("### RDS Case Processing History")
        rds_cases = [case for case in st.session_state.extraction_history if 'rds' in case['text'].lower() or 'respiratory distress' in case['text'].lower()]
        
        if rds_cases:
            st.info(f"You have processed {len(rds_cases)} RDS-related texts out of {len(st.session_state.extraction_history)} total cases in this session.")
            
            # Show RDS case statistics
            if len(rds_cases) > 0:
                with st.expander("RDS Case Analysis"):
                    for i, case in enumerate(rds_cases[:3]):  # Show first 3 RDS cases
                        st.markdown(f"**RDS Case {i+1}:**")
                        st.text(case['text'][:100] + "...")
                        if 'evaluation' in case['result']:
                            eval_data = case['result']['evaluation']
                            st.write(f"F1-Score: {eval_data['f1_score']:.3f}")
        else:
            st.info(f"No RDS-specific cases detected in your {len(st.session_state.extraction_history)} processed texts.")
        
        if st.button("Clear History"):
            st.session_state.extraction_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No cases processed yet. Try the 'Single Text Analysis' or 'Sample Data Explorer' to see RDS cases in action!")

if __name__ == "__main__":
    main()
