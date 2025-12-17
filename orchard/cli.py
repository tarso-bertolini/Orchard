import argparse
import sys
import subprocess
import platform
import os
from typing import List, Dict

def get_system_info():
    """Retrieves system information (RAM, Chip) on macOS."""
    info = {}
    
    # Get Chip Name
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        info["chip"] = result.stdout.strip()
    except Exception:
        info["chip"] = "Unknown Apple Silicon"

    # Get RAM
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        bytes_ram = int(result.stdout.strip())
        gb_ram = bytes_ram / (1024**3)
        info["ram"] = f"{gb_ram:.1f} GB"
        info["ram_bytes"] = bytes_ram
    except Exception:
        info["ram"] = "Unknown"
        info["ram_bytes"] = 0

    info["os"] = f"{platform.system()} {platform.release()}"
    return info

def recommend_models(ram_bytes: int) -> List[Dict[str, str]]:
    """Recommends models based on available RAM."""
    # Rough estimates for 4-bit quantized models
    # 7B ~ 4-5GB
    # 13B ~ 8-9GB
    # 70B ~ 40GB
    
    models = [
        {"name": "TinyLlama-1.1B-Chat-v1.0", "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size_gb": 1.0},
        {"name": "Llama-2-7b-chat", "id": "meta-llama/Llama-2-7b-chat-hf", "size_gb": 5.0},
        {"name": "Mistral-7B-Instruct-v0.2", "id": "mistralai/Mistral-7B-Instruct-v0.2", "size_gb": 5.0},
        {"name": "Llama-2-13b-chat", "id": "meta-llama/Llama-2-13b-chat-hf", "size_gb": 9.0},
        {"name": "Mixtral-8x7B-Instruct-v0.1", "id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size_gb": 26.0},
    ]
    
    # Filter based on RAM (leave 2GB for system)
    available_ram_gb = ram_bytes / (1024**3)
    safe_limit = available_ram_gb - 2.0
    
    recommended = []
    for m in models:
        status = "✅ Recommended" if m["size_gb"] <= safe_limit else "⚠️  May swap/crash"
        if m["size_gb"] > available_ram_gb:
            status = "❌ Insufficient RAM"
        
        m["status"] = status
        recommended.append(m)
        
    return recommended

def download_model_interactive():
    print("\n--- System Check ---")
    sys_info = get_system_info()
    print(f"Chip: {sys_info['chip']}")
    print(f"RAM:  {sys_info['ram']}")
    
    print("\n--- Available Models ---")
    models = recommend_models(sys_info.get("ram_bytes", 0))
    
    for idx, m in enumerate(models):
        print(f"{idx + 1}. {m['name']:<30} | ~{m['size_gb']} GB | {m['status']}")
    
    print("\nSelect a model to download (enter number) or 'q' to quit:")
    choice = input("> ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected = models[idx]
            print(f"\nDownloading {selected['name']} from Hugging Face...")
            print(f"Repo ID: {selected['id']}")
            
            # Use huggingface_hub to download
            try:
                from huggingface_hub import snapshot_download
                local_dir = f"models/{selected['name']}"
                print(f"Saving to {local_dir}...")
                snapshot_download(repo_id=selected['id'], local_dir=local_dir, local_dir_use_symlinks=False)
                print(f"\n✅ Download complete! You can run it with:\n orchard run --model {local_dir}")
            except ImportError:
                print("Error: 'huggingface_hub' is not installed. Please install it via pip.")
                print("pip install huggingface_hub")
            except Exception as e:
                print(f"Error downloading: {e}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

def main():
    parser = argparse.ArgumentParser(description="Orchard CLI - Apple Silicon LLM Runtime")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Info command
    subparsers.add_parser("info", help="Show system information")
    
    # Download command
    subparsers.add_parser("download", help="Download a model")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    run_parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Initial prompt")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize (quantize) a model for faster loading")
    optimize_parser.add_argument("--model", type=str, required=True, help="Path to source model directory")
    optimize_parser.add_argument("--output", type=str, required=True, help="Path to output directory")

    args = parser.parse_args()
    
    if args.command == "info":
        info = get_system_info()
        print(f"Orchard Runtime v0.1.0")
        print(f"----------------------")
        print(f"OS:   {info['os']}")
        print(f"Chip: {info['chip']}")
        print(f"RAM:  {info['ram']}")
        
    elif args.command == "download":
        download_model_interactive()
        
    elif args.command == "optimize":
        try:
            from orchard.optimizer import optimize_model
            optimize_model(args.model, args.output)
        except Exception as e:
            print(f"Error optimizing model: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == "run":
        print(f"Loading model from {args.model}...")
        try:
            from orchard.model import Llama
            model = Llama(args.model)
            print(f"\nGenerating response to: '{args.prompt}'\n")
            model.generate(args.prompt)
        except Exception as e:
            print(f"Error running model: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
