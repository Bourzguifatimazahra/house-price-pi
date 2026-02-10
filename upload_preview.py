import gradio as gr
import pandas as pd
import numpy as np

def preview_file(file):
    if file is None:
        return "Please upload a file."
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        else:
            return "Unsupported format. Use CSV or Excel."
        
        return df.head()
    
    except Exception as e:
        return f"Error: {e}"

def analyze_file(file):
    if file is None:
        return "Please upload a file."
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        else:
            return "Unsupported format."
        
        analysis = f"**File Analysis:**\n\n"
        analysis += f"• Rows: {len(df):,}\n"
        analysis += f"• Columns: {len(df.columns):,}\n"
        analysis += f"• Memory: {df.memory_usage().sum() / 1024 / 1024:.2f} MB\n\n"
        
        analysis += "**Data Types:**\n"
        for dtype in df.dtypes.unique():
            count = (df.dtypes == dtype).sum()
            analysis += f"• {dtype}: {count} columns\n"
        
        analysis += "\n**Missing Values:**\n"
        missing = df.isnull().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                analysis += f"• {col}: {count:,} ({pct:.1f}%)\n"
        else:
            analysis += "• None\n"
        
        return analysis
    
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="WA Real Estate Data Preview") as demo:
    gr.Markdown("# 📁 Washington Real Estate Data Preview")
    
    with gr.Tab("Preview"):
        file_input = gr.File(label="Upload CSV or Excel")
        preview_btn = gr.Button("Preview")
        preview_output = gr.Dataframe(label="First 5 Rows")
        preview_btn.click(preview_file, inputs=file_input, outputs=preview_output)
    
    with gr.Tab("Analysis"):
        file_input2 = gr.File(label="Upload CSV or Excel")
        analyze_btn = gr.Button("Analyze")
        analysis_output = gr.Markdown(label="Analysis Results")
        analyze_btn.click(analyze_file, inputs=file_input2, outputs=analysis_output)

if __name__ == "__main__":
    demo.launch()