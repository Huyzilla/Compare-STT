import pandas as pd
import string
import jiwer
import os
import argparse
import glob

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

def calculate_metrics(df, hyp_col, ref_col='ground_truth'):
    if hyp_col not in df.columns:
        return 0, 0
    
    hyps = df[hyp_col].apply(normalize_text).tolist()
    refs = df[ref_col].apply(normalize_text).tolist()
    
    valid_indices = [i for i, ref in enumerate(refs) if len(ref) > 0]
    if not valid_indices:
        return 0, 0
    
    hyps = [hyps[i] for i in valid_indices]
    refs = [refs[i] for i in valid_indices]
    
    try:
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
    except Exception:
        wer, cer = 0, 0
    
    return wer, cer

def find_transcription_column(columns):
    """Attempts to find the transcription column automatically."""
    for col in columns:
        if '_transcription' in col:
            return col
    # Fallback for older versions or manually edited files
    for col in ['gemini', 'iflytek', 'google']:
        if col in columns:
            return col
    return None

def main():
    parser = argparse.ArgumentParser(description="Aggregate ASR Benchmark Results")
    parser.add_argument("--dir", type=str, default="output/dolly", help="Directory containing result CSV files")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    csv_files = glob.glob(os.path.join(args.dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    results = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Derive model name: remove _results.csv or results.csv
        model_name = filename.replace("_results.csv", "").replace("results.csv", "").replace("_", " ").title()
        
        try:
            df = pd.read_csv(file_path)
            gt_col = 'ground_truth' if 'ground_truth' in df.columns else 'gt'
            
            if gt_col not in df.columns:
                print(f"Skipping {filename}: Missing ground truth column.")
                continue
                
            trans_col = find_transcription_column(df.columns)
            if not trans_col:
                print(f"Skipping {filename}: Could not find transcription column.")
                continue
            
            wer, cer = calculate_metrics(df, trans_col, gt_col)
            
            # Timing metrics
            avg_proc_time = df['processing_time'].mean() if 'processing_time' in df.columns else 0
            avg_rtf = df['rtf'].mean() if 'rtf' in df.columns else 0
            
            results.append({
                'Model': model_name,
                'WER (%)': f"{wer*100:.2f}%",
                'CER (%)': f"{cer*100:.2f}%",
                'Avg Time (s)': f"{avg_proc_time:.2f}s",
                'Avg RTF': f"{avg_rtf:.4f}",
                'Samples': len(df)
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by='WER (%)')
        print("\n" + "="*80)
        print(f"                ASR BENCHMARK SUMMARY (Folder: {args.dir})")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        print("Metrics: WER (Word Error Rate), CER (Character Error Rate), RTF (Real-Time Factor)")
        print("Note: Models are sorted by accuracy (WER).")
    else:
        print("No valid benchmark results found.")

if __name__ == "__main__":
    main()
