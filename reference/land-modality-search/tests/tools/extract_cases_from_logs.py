import json
import sys
import os

def extract_cases(log_file_path, output_dir):
    try:
        with open(log_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {log_file_path}: {e}")
        return

    def process_node(node, current_path=""):
        if isinstance(node, dict):
            # Check if this node is a run result (has "Full raw LLM response")
            response_text = node.get("Full raw LLM response")
            if response_text:
                extracted_grid = node.get("Extracted grid")
                
                # Sanitize key for filename
                safe_key = "".join([c if c.isalnum() or c in "._-" else "_" for c in current_path])
                if not safe_key:
                    safe_key = "root"
                
                out_path = os.path.join(output_dir, f"{safe_key}.txt")
                with open(out_path, "w") as f:
                    f.write(response_text)
                print(f"Saved case: {out_path}")
            
            # Recurse into children
            for key, value in node.items():
                new_path = f"{current_path}_{key}" if current_path else key
                process_node(value, new_path)
    
    process_node(data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract LLM responses from log files for regression testing.")
    parser.add_argument("log_files", nargs="+", help="Path(s) to JSON log files.")
    parser.add_argument("--output-dir", default="tests/grid_parsing_cases/cases", help="Directory to save extracted text files.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    for log_file in args.log_files:
        if os.path.exists(log_file):
            print(f"Extracting from: {log_file}")
            extract_cases(log_file, args.output_dir)
        else:
            print(f"Skipping missing file: {log_file}")
