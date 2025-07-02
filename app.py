import gradio as gr
import pandas as pd
import numpy as np
import random
import tempfile
import os
import zipfile
import io
from Bio import SeqIO
import torch
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go


class RPContactPredictor:
    def __init__(self, model_path='./weight/model_roc_0_56=0.779.pt'):

        """Initialize RNA-protein contact predictor"""
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.seed_everything()

    def seed_everything(self, seed=2022):
        """Set random seed for reproducibility"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def one_hot_encode(self, sequences, alpha='ACGU'):
        """One-hot encode biological sequences"""
        sequences_array = np.array(list(sequences)).reshape(-1, 1)
        label = np.array(list(alpha)).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(label)
        seq_encode = enc.transform(sequences_array).toarray()
        return seq_encode

    def contact_partner_constrained(self, prob_matrix, colmax=12, rowmax=24):
        """Apply contact partner constraints to probability matrix"""
        row_max_indices = np.argsort(-prob_matrix, axis=1)[:, :rowmax]
        row_max_mask = np.zeros_like(prob_matrix)
        row_max_mask[np.arange(prob_matrix.shape[0])[:, np.newaxis], row_max_indices] = 1

        col_max_indices = np.argsort(-prob_matrix, axis=0)[:colmax, :]
        col_max_mask = np.zeros_like(prob_matrix)
        col_max_mask[col_max_indices, np.arange(prob_matrix.shape[1])] = 1

        mask = np.logical_and(row_max_mask, col_max_mask).astype(np.float32)
        prob_matrix = np.where(mask == 1, prob_matrix, 0)
        return prob_matrix

    def read_fasta(self, fasta_content):
        """Parse FASTA format content"""
        sequences = {}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            tmp_file.write(fasta_content)
            tmp_file_path = tmp_file.name

        try:
            for record in SeqIO.parse(tmp_file_path, 'fasta'):
                pdbid, seq = record.id, str(record.seq)
                rnaid, proid = pdbid.split('.')
                rnaseq, proseq = seq.split('.')
                sequences = {
                    'rna': (rnaid, rnaseq),
                    'protein': (proid, proseq)
                }
                break
        finally:
            os.unlink(tmp_file_path)

        return sequences

    def predict_contact(self, rna_seq, protein_seq):
        """Predict RNA-protein contact matrix"""
        # Encode sequences
        rna_oh = self.one_hot_encode(rna_seq, alpha='ACGU')
        pro_oh = self.one_hot_encode(protein_seq, alpha='GAVLIFWYDNEKQMSTCPHR')

        # Prepare input tensors
        x_rna = torch.from_numpy(np.expand_dims(rna_oh, 0)).transpose(-1, -2).float()
        x_pro = torch.from_numpy(np.expand_dims(pro_oh, 0)).transpose(-1, -2).float()

        # Run prediction
        with torch.no_grad():
            outputs = self.model(x_pro, x_rna)

        # Process outputs
        outputs = torch.squeeze(outputs, -1).permute(0, 2, 1)
        contact_matrix = outputs[0].cpu().numpy()

        # Apply constraints and normalization
        contact_matrix = self.contact_partner_constrained(contact_matrix)
        contact_matrix = (contact_matrix - contact_matrix.min()) / (contact_matrix.max() - contact_matrix.min() + 1e-8)

        return contact_matrix


def create_heatmap(contact_matrix, rna_labels, protein_labels, rna_name, protein_name, Threshold=0.0):
    """Create interactive contact heatmap with threshold filtering"""
    # Apply Threshold threshold
    filtered_matrix = contact_matrix.copy()
    filtered_matrix[filtered_matrix < Threshold] = 0

    fig = go.Figure(data=go.Heatmap(
        z=filtered_matrix,
        x=protein_labels,
        y=rna_labels,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Predicted Probability"),
        hovertemplate='RNA: %{y}<br>Protein: %{x}<br>Probability: %{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f"{rna_name} vs {protein_name} (Threshold ≥ {Threshold:.3f})",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f"Protein Residues ({protein_name})",
        yaxis_title=f"RNA Nucleotides ({rna_name})",
        width=800,
        height=600,
        font=dict(size=12)
    )

    return fig


def get_contact_pairs(contact_matrix, rna_labels, protein_labels, Threshold=0.0):
    """Get filtered contact pairs list above threshold"""
    df = pd.DataFrame(contact_matrix, index=rna_labels, columns=protein_labels)
    df_stacked = df.stack().reset_index()
    df_stacked.columns = ['RNA', 'Protein', 'Probability']
    df_filtered = df_stacked[df_stacked['Probability'] > Threshold].sort_values('Probability', ascending=False)
    return df_filtered


def create_download_files(contact_matrix, rna_labels, protein_labels, rna_name, protein_name):
    """Create downloadable result files package"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save heatmap raw data
    heatmap_df = pd.DataFrame(contact_matrix, index=rna_labels, columns=protein_labels)
    heatmap_file = os.path.join(temp_dir, f"{rna_name}_{protein_name}_heatmap.csv")
    heatmap_df.to_csv(heatmap_file, index=True)

    # Save contact pairs list
    pairs_df = get_contact_pairs(contact_matrix, rna_labels, protein_labels, Threshold=0.0)
    pairs_file = os.path.join(temp_dir, f"{rna_name}_{protein_name}_contact_pairs.csv")
    pairs_df.to_csv(pairs_file, index=False)

    # Create ZIP file
    zip_path = os.path.join(temp_dir, f"{rna_name}_{protein_name}_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(heatmap_file, os.path.basename(heatmap_file))
        zipf.write(pairs_file, os.path.basename(pairs_file))

    return zip_path


def process_prediction(fasta_file, rna_sequence, protein_sequence, input_method):
    """Process prediction request and return initial results"""
    if not fasta_file and not (rna_sequence and protein_sequence):
        return "❌ Please upload a FASTA file or enter RNA and protein sequences",None, None, None, None, None, None

    try:
        # Process input
        if input_method == "Upload FASTA File" and fasta_file:
            fasta_content = fasta_file.decode('utf-8')
            sequences = predictor.read_fasta(fasta_content)
        else:
            # Create sequences from text input
            sequences = {
                'rna': ('RNA', rna_sequence),
                'protein': ('Protein', protein_sequence)
            }

        rna_id, rna_seq = sequences['rna']
        protein_id, protein_seq = sequences['protein']

        # Validate sequences
        if len(set(rna_seq) - set('ACGU')) > 0:
            return f"❌ RNA sequence contains invalid characters: {set(rna_seq) - set('ACGU')}",None, None, None, None, None, None
        if len(set(protein_seq) - set('GAVLIFWYDNEKQMSTCPHR')) > 0:
            return f"❌ Protein sequence contains invalid characters: {set(protein_seq) - set('GAVLIFWYDNEKQMSTCPHR')}",None, None, None, None, None, None

        # Run contact prediction
        contact_matrix = predictor.predict_contact(rna_seq, protein_seq)

        # Generate residue labels
        rna_labels = [f'{nt}{i + 1}' for i, nt in enumerate(rna_seq)]
        protein_labels = [f'{aa}{i + 1}' for i, aa in enumerate(protein_seq)]

        # Calculate default Threshold (minimum non-zero value)
        non_zero_values = contact_matrix[contact_matrix > 0]
        default_threshold = float(np.min(non_zero_values)) if len(non_zero_values) > 0 else 0.0
        max_threshold = float(np.max(contact_matrix))

        # Create initial heatmap with default Threshold
        heatmap = create_heatmap(contact_matrix, rna_labels, protein_labels, rna_id, protein_id, default_threshold)

        # Create initial contact pairs table
        contact_pairs = get_contact_pairs(contact_matrix, rna_labels, protein_labels, default_threshold)

        # Create download file
        download_file = create_download_files(contact_matrix, rna_labels, protein_labels, rna_id, protein_id)

        # Prepare status message
        status = f"✅ Prediction completed!\n"
        status += f"RNA length: {len(rna_seq)}\n"
        status += f"Protein length: {len(protein_seq)}\n"
        status += f"Total predicted contacts: {len(contact_pairs)}"

        # Prepare result state for threshold updates
        result_state = {
            'contact_matrix': contact_matrix,
            'rna_labels': rna_labels,
            'protein_labels': protein_labels,
            'rna_id': rna_id,
            'protein_id': protein_id
        }

        # Update slider configuration
        slider_update = gr.update(
            minimum=default_threshold,
            maximum=max_threshold,
            value=default_threshold,
            step=(max_threshold - default_threshold) / 100,
            visible=True
        )

        # Create contact pairs info
        contact_info = f"📊 Found {len(contact_pairs)} contacts (Threshold ≥ {default_threshold:.3f})"

        return status, heatmap, contact_pairs, contact_info, download_file, result_state, slider_update

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}", None, None, None, None, None, None

def update_results_with_threshold(Threshold, result_state):
    """Update heatmap and contact table based on Threshold threshold"""
    if result_state is None:
        return None, None, None
    # Create updated heatmap
    heatmap = create_heatmap(
        result_state['contact_matrix'],
        result_state['rna_labels'],
        result_state['protein_labels'],
        result_state['rna_id'],
        result_state['protein_id'],
        Threshold
    )

    # Create updated contact pairs table
    contact_pairs = get_contact_pairs(
        result_state['contact_matrix'],
        result_state['rna_labels'],
        result_state['protein_labels'],
        Threshold
    )

    # Create contact pairs info
    contact_info = f"📊 Found {len(contact_pairs)} contacts (Probability ≥ {Threshold:.3f})"


    return heatmap, contact_pairs, contact_info


def reset_threshold(result_state):
    if result_state is None:
        return gr.update(value=0.0)

    contact_matrix = result_state['contact_matrix']
    non_zero_values = contact_matrix[contact_matrix > 0]

    if len(non_zero_values) > 0:
        default_threshold = float(np.min(non_zero_values))
    else:
        default_threshold = 0.0

    # 返回滑块更新对象
    return gr.update(
        minimum=default_threshold,
        maximum=float(np.max(non_zero_values)),
        value=default_threshold,
        interactive=True)


def load_example_data(fasta_input, rna_input, protein_input):
    # 如果fasta有值（非空），则返回"Upload FASTA File"
    if fasta_input is not None:
        return gr.update(value="Upload FASTA File")
    else:
        return gr.update(value="Enter Sequences Directly")
def create_interface():
    """Create Gradio interface with threshold control"""
    custom_css = """
    .gradio-dataframe {
        background: white !important;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .dataframe-container {
        padding: 12px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .contact-info {
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
        color: #4a5568;
    }
    """

    with gr.Blocks(title="RNA-Protein Contact Prediction Tool",
                   theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
                   css=custom_css) as app:
        gr.Markdown("""
   <center>
   
# 🧬 RPcontact: RNA-Protein Contact Prediction  
**Direct Nucleotide–Residue Contact Prediction from Primary Sequences**

[Paper](https://www.biorxiv.org/content/10.1101/2025.06.02.657171v1.full) | 
[Code](https://github.com/rpcontact) | 
[Demo](https://julse-rpcontact.hf.space/)

</center>


> RPcontact predicts direct nucleotide-residue contacts between RNA and protein sequences. 
Leveraging **ERNIE-RNA** for RNA and **ESM-2** for protein modeling, the method provides high-resolution insights into RNA-protein interactions at the atomic level.
<br><br>Current Demo (auROC 0.779 on VL-49) is optimized for limited CPU environments using efficient one-hot encoding<br>
    Advanced Model (auROC 0.845 on VL-49), the Embedding-based approach will be released upon paper publication ([contact us](mailto:jiangjh2023@shanghaitech.edu.cn) for early access)
        
        """)
        with gr.Tab("🔬 Contact Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ Input Options")
                    with gr.Group(elem_classes="input-group"):
                        input_method = gr.Radio(
                            choices=["Upload FASTA File", "Enter Sequences Directly"],
                            value="Upload FASTA File",
                            label="Input Method"
                        )

                        fasta_input = gr.File(
                            label="FASTA File",
                            file_types=['.fasta', '.fa', '.txt'],
                            type='binary'
                        )

                        rna_input = gr.Textbox(
                            label="RNA Sequence",
                            placeholder="Enter RNA sequence (use A,C,G,U)",
                            lines=3,
                            visible=False
                        )

                        protein_input = gr.Textbox(
                            label="Protein Sequence",
                            placeholder="Enter protein sequence (standard amino acid codes)",
                            lines=3,
                            visible=False
                        )

                    # Example data
                    gr.Examples(
                        examples=[
                            ["./example/inputs/8DMB_W.8DMB_P.fasta", "GGGCCUUAUUAAAUGACUUC", "MDVPRKMETRRNLRRARRYRK"],
                        ],
                        inputs=[fasta_input, rna_input, protein_input],
                        outputs=[input_method],
                        label="📋 Example Data (click to load)",
                        run_on_click=True,
                        fn = load_example_data
                    )



                    # Submit button at the bottom of input column
                    predict_btn = gr.Button("🚀 Run Prediction", variant="primary", size="lg")

                    # Status output
                    status_output = gr.Textbox(label="Prediction Status", lines=5)



                with gr.Column(scale=2):
                    # Results section - initially hidden
                    gr.Markdown("""
                    ## 📊 Results
                    """)
                    # Threshold control section
                    with gr.Row():
                        threshold_slider = gr.Slider(
                            label="Contact Probability Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.001,
                            visible=True,
                            interactive=True
                        )
                        reset_btn = gr.Button("Reset to Default", size="sm")
                    gr.Markdown("""
                    ### 🎯Contact Map
                    """)
                    # Heatmap display
                    heatmap_plot = gr.Plot(label='Contact Map')

                    # Contact pairs table with info header
                    gr.Markdown("### 🎯Contact Pairs")
                    contact_info = gr.Markdown("", elem_classes="contact-info")
                    contact_table = gr.Dataframe(
                        headers=["RNA", "Protein", "Probability"],
                        datatype=["str", "str", "number"],
                        row_count=15,
                        interactive=False,
                        elem_classes="gradio-dataframe"
                    )

                    # Download button
                    download_btn = gr.File(
                        label="📥 Download Results Package",
                        visible=True
                    )

        # User Guide tab remains unchanged
        with gr.Tab("📖 User Guide"):
            # ... (unchanged user guide content) ...
            gr.Markdown("""
            # 📖 Comprehensive User Guide

            ## 🎯 Overview

            This tool predicts direct contacts between nucleotides in RNA sequences and residues in protein sequences using a deep learning model based on ERNIE-RNA and ESM-2 embeddings. The tool provides:

            - **Interactive contact matrix visualization** with adjustable probability thresholds
            - **Detailed contact pairs list** sorted by prediction confidence
            - **Downloadable results** in CSV and ZIP formats
            - **Real-time threshold filtering** for result exploration

            ## 📋 Input Formats

            ### Method 1: FASTA File Upload

            Upload a FASTA file containing both RNA and protein sequences in the following format:

            ```
            >RNA_ID.PROTEIN_ID
            RNA_SEQUENCE.PROTEIN_SEQUENCE
            ```

            **Example:**
            ```
            >8DMB_W.8DMB_P
            GGGCCUUAUUAAAUGACUUC.MDVPRKMETRRNLRRARRYRK
            ```

            ### Method 2: Direct Sequence Input

            Enter RNA and protein sequences directly in the respective text boxes:

            - **RNA Sequence**: Use standard nucleotide codes (A, U, G, C)
            - **Protein Sequence**: Use standard single-letter amino acid codes (GAVLIFWYDNEKQMSTCPHR)

            ## 🔬 Understanding Results

            ### Contact Heatmap

            - **X-axis**: Protein residue positions (e.g., M1, D2, V3...)
            - **Y-axis**: RNA nucleotide positions (e.g., G1, G2, G3...)
            - **Color Intensity**: Contact probability (0.0 to 1.0)
            - **Red Colors**: Higher contact probability
            - **White/Light**: Lower or no contact probability

            ### Contact Pairs Table

            Lists all predicted contacts above the selected threshold, showing:
            - **RNA**: Nucleotide position and type
            - **Protein**: Residue position and type  
            - **Probability**: Contact prediction confidence (0.0-1.0)

            ### Threshold Control

            Use the **Contact Probability Threshold** slider to:
            - Filter contacts by minimum probability
            - Focus on high-confidence predictions
            - Explore different confidence levels
            - Click **"Reset to Default"** to return to the minimum non-zero value

            ## 📥 Download Options

            The results package (ZIP file) contains:

            1. **`*_heatmap.csv`**: Complete contact probability matrix
                - Rows: RNA nucleotides  
                - Columns: Protein residues
                - Values: Contact probabilities

            2. **`*_contact_pairs.csv`**: All contact pairs above zero probability
                - RNA: Nucleotide identifier
                - Protein: Residue identifier
                - Probability: Contact prediction score

            ## ⚡ Performance Guidelines

            - **Processing Time**: Scales quadratically with sequence length
            
            ### Quality Considerations
            - Higher probabilities indicate more confident predictions
            - Consider biological context when interpreting results
            - Cross-validate important contacts with experimental data

            ## 🔧 Troubleshooting

            ### Common Issues

            **Invalid Characters Error:**
            - RNA: Only A, U, G, C are allowed
            - Protein: Only standard 20 amino acids are supported
            - Check for lowercase letters, numbers, or special characters

            **File Format Error:**
            - Ensure FASTA format: `>ID\\nSEQUENCE`
            - Use period (.) to separate RNA and protein sequences
            - Check file encoding (UTF-8 recommended)

            **Empty Results:**
            - Very short sequences may produce no significant contacts
            - Try lowering the probability threshold
            - Verify sequence quality and biological relevance

            ## 📊 Interpretation Guidelines

            ### High-Confidence Predictions (≥0.7)
            - Strong likelihood of direct contact
            - Priority targets for experimental validation
            - Suitable for structural modeling constraints

            ### Medium-Confidence Predictions (0.3-0.7)
            - Moderate likelihood of interaction
            - Consider in context with other evidence
            - Useful for identifying interaction regions

            ### Low-Confidence Predictions (<0.3)
            - May represent weak or indirect interactions
            - Use with caution for biological interpretation
            - Good for exploratory analysis

            ## 🔬 Technical Details

            ### Model Architecture
            - Based on attention mechanisms and transformer models
            - Trained on experimentally validated RNA-protein complexes
            - Uses one-hot encoding for sequence representation
            - Applies contact partner constraints for biological realism

            ### Validation Metrics
            - Cross-validated on diverse RNA-protein complex datasets
            - Performance metrics available in the original publication
            - Benchmarked against existing prediction methods
            
            ### 📊 Difference between current demo and final model
            | Model Type          | Checkpoint File           | auROC (VL-49) | LLM embeddings |
            |---------------------|---------------------------|---------------|-------------------|
            | OH + RP_Emb (final)         | `model_roc_0_38=0.845.pt` | 0.845         | ✓                |
            | OH (demo)                  | `model_roc_0_56=0.779.pt` | 0.779         | ✗                |
            
            ## 📚 Citation & Contact

            If you use this tool in your research, please cite:

            **Jiang, J., Zhang, X., Zhan, J., Miao, Z., & Zhou, Y. (2025). RPcontact: Improved prediction of RNA-protein contacts using RNA and protein language models. bioRxiv, 2025-06.**

            ### Contact Information
            For technical issues, feature requests, or collaboration inquiries, please contact the development team.
            
            - **Primary Contact**: Jiuhong Jiang
            - **Email**: jiangjh2023@shanghaitech.edu.cn
            - **Institution**: ShanghaiTech University, Shanghai, China
            ---

            <p align="center"><em>Making RNA-protein interaction prediction accessible and accurate for the research community.</em></p>

            """)

        # Hidden state to store prediction results
        result_state = gr.State()

        # Event handlers
        def toggle_inputs(method):
            """Toggle input visibility based on selected method"""
            if method == "Upload FASTA File":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

        # Input method change
        input_method.change(
            fn=toggle_inputs,
            inputs=[input_method],
            outputs=[fasta_input, rna_input, protein_input]
        )

        # Prediction button
        predict_btn.click(
            fn=process_prediction,
            inputs=[fasta_input, rna_input, protein_input, input_method],
            outputs=[
                status_output,
                heatmap_plot,
                contact_table,
                contact_info,
                download_btn,
                result_state,
                threshold_slider
            ]
        )

        # Threshold slider change
        threshold_slider.change(
            fn=update_results_with_threshold,
            inputs=[threshold_slider, result_state],
            outputs=[heatmap_plot, contact_table, contact_info]
        )

        # Reset button
        reset_btn.click(
            fn=reset_threshold,
            inputs=[result_state],
            outputs=[threshold_slider]
        )

    return app


# Initialize predictor
predictor = RPContactPredictor()

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )