"""
TensorRT Preprocessing Model Export Script

converts it to a TensorRT engine using trtexec.

Usage:
    python export_preprocess_trt.py [--num_windows 3] [--max_batch_size 4] [--target_size 512]
"""

import os
import sys
import argparse
import subprocess
import torch
import torch.onnx
from pathlib import Path

# Add current directory to path to import preprocess_module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_module import TAO_DDETRPreprocessModule


def export_to_onnx(model, batch_size, input_height, input_width, output_path, opset_version=18):
    """
    Export PyTorch model to ONNX format with proper dynamic axes.
    
    Args:
        model: PyTorch model to export
        batch_size: Batch size for export
        input_height: Input image height
        input_width: Input image width  
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (17 for better TensorRT support)
    """
    print(f"Exporting model to ONNX: {output_path}")
    
    # Create dummy input with optimal batch size in NCHW format
    # Use UINT8 for ONNX export (no more Gather operations to cause issues)
    if input_height == -1:
            input_height = 1080
    if input_width == -1:
        input_width = 1920
    dummy_input = torch.randint(0, 255, (batch_size, 3, input_height, input_width), dtype=torch.uint8)
    
    # Set model to eval mode
    model.eval()
    
    # Define dynamic axes for height/width only (batch size fixed at 1)
    dynamic_axes = {
        'input_images': {2: 'input_height', 3: 'input_width'},  # Dynamic height/width only
        'preprocessed_image': {}  # Fixed shape output
    }
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_images'],
        output_names=['preprocessed_image'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"ONNX export completed: {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    except ImportError:
        print("Warning: onnx package not available for verification")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")


def convert_to_tensorrt(onnx_path, engine_path, max_batch_size, input_height, input_width, 
                       target_width, target_height, fp16=True, workspace_size=2048):
    """
    Convert ONNX model to TensorRT engine using trtexec with proper batch configuration.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        max_batch_size: Maximum batch size to support (1-4 cameras)
        input_height, input_width: Input image dimensions
        num_windows: Number of windows per image
        target_size: Target size for each window tile
        fp16: Whether to use FP16 precision
        workspace_size: Workspace size in MB
    """
    print(f"Converting ONNX to TensorRT engine: {engine_path}")
    print(f"Batch configuration: {max_batch_size}")
    print(f"Frame configuration: {target_height}x{target_width} each")
    

    if input_height == -1:
        min_input_height = 1
        opt_input_height = 480
        max_input_height = 1080
    else:
        min_input_height = input_height
        opt_input_height = input_height
        max_input_height = input_height
    if input_width == -1:
        min_input_width = 1
        opt_input_width = 640
        max_input_width = 1920
    else:
        min_input_width = input_width
        opt_input_width = input_width
        max_input_width = input_width
        
        
    # Construct trtexec command with proper shapes
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_size}",  # Updated for newer TensorRT versions
        # Input/Output format specifications for UINT8 support
        "--inputIOFormats=uint8:chw",
        "--outputIOFormats=fp32:chw",
        # Fixed batch size of 1 - NCHW format
        f"--minShapes=input_images:1x3x{min_input_height}x{min_input_width}",
        f"--optShapes=input_images:1x3x{opt_input_height}x{opt_input_width}",
        f"--maxShapes=input_images:1x3x{max_input_height}x{max_input_width}",
        # Enable optimizations
        "--builderOptimizationLevel=5",
        "--avgTiming=100",
        "--verbose"
    ]
    
    if fp16:
        cmd.append("--fp16")
        cmd.append("--best")  # Let TensorRT choose the best precision
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run trtexec
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ TensorRT engine conversion successful!")
        print(f"TensorRT engine saved to: {engine_path}")
        
        # Print performance summary
        if "mean:" in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if "mean:" in line.lower() and "ms" in line.lower():
                    print(f"Performance: {line.strip()}")
                    
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during TensorRT conversion: {e}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("--- stdout ---")
            print(e.stdout[-2000:])  # Print last 2000 chars
        if e.stderr:
            print("--- stderr ---")
            print(e.stderr[-2000:])
        raise
    except FileNotFoundError:
        print("✗ Error: trtexec not found. Please ensure TensorRT is installed and trtexec is in PATH.")
        print("Installation guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/")
        raise


def validate_model(model, batch_size, input_height, input_width, target_width, target_height):
    """
    Validate the model with test inputs of different batch sizes.
    """
    print("Validating model...")
    
    # Test batch size 1 only
    for test_batch_size in [1]:
        print(f"\nTesting batch size: {test_batch_size}")
        
        # Create test input in NCHW format
        if input_height == -1:
            input_height = 1080
        if input_width == -1:
            input_width = 1920
        test_input = torch.randint(0, 256, (test_batch_size, 3, input_height, input_width), dtype=torch.uint8)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Validate output shape - always flattened: [batch_size * num_windows, 3, target_size, target_size]
        expected_shape = (test_batch_size, 3, target_height, target_width)
        if output.shape != expected_shape:
            print(f"  ✗ Warning: Expected output shape {expected_shape}, got {output.shape}")
        else:
            print(f"  ✓ Output shape validation passed")
        
        # Validate output range (should be normalized)
        if output.min() < -5 or output.max() > 5:
            print(f"  ✗ Warning: Output values seem out of expected normalized range")
        else:
            print(f"  ✓ Output normalization validation passed")


def generate_model_name(max_batch_size, target_width, target_height, input_width, input_height):
    """Generate descriptive model name based on configuration."""
    return f"ddter_{max_batch_size}batch_{target_width}tw_{target_height}th_{input_width}w_{input_height}h"


def main():
    parser = argparse.ArgumentParser(description='Export preprocessing module to TensorRT')
    parser.add_argument('--input_height', type=int, default=-1,
                       help='Input image height (default: -1)')
    parser.add_argument('--input_width', type=int, default=-1,
                       help='Input image width (default: -1)')
    parser.add_argument('--target_width', type=int, default=960,
                       help='Target tile size after resizing (default: 960)')
    parser.add_argument('--target_height', type=int, default=544,
                       help='Target tile size after resizing (default: 544)')
    parser.add_argument('--max_batch_size', type=int, default=1,
                       help='Maximum batch size to support (default: 1)')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Use FP16 precision (default: False)')
    parser.add_argument('--workspace', type=int, default=2048,
                       help='TensorRT workspace size in MB (default: 2048)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate descriptive model names
    base_name = generate_model_name(args.max_batch_size, args.target_width, args.target_height, args.input_width, args.input_height)
    onnx_path = output_dir / f"{base_name}.onnx"
    engine_path = output_dir / f"{base_name}.plan"
    
    print("="*80)
    print("DDETR Preprocessing Module Export")
    print("="*80)
    print(f"Configuration:")
    print(f"  Input shape: {args.max_batch_size}x{args.input_height}x{args.input_width}x3")
    print(f"  Target tile size: {args.target_height}x{args.target_width}")
    print(f"  Batch size: {args.max_batch_size}")
    print(f"  FP16 precision: {args.fp16}")
    print(f"  Workspace size: {args.workspace} MB")
    print(f"Output files:")
    print(f"  ONNX: {onnx_path}")
    print(f"  TensorRT: {engine_path}")
    print("="*80)
    
    try:
        # Create model with specified configuration
        model = TAO_DDETRPreprocessModule(
            target_width=args.target_width,
            target_height=args.target_height
        )
        
        # Validate model
        validate_model(model, args.max_batch_size, args.input_height, args.input_width, 
                      args.target_width, args.target_height)
        
        # Export to ONNX
        export_to_onnx(
            model, 
            args.max_batch_size,
            args.input_height, 
            args.input_width,
            str(onnx_path)
        )
        
        # Convert to TensorRT
        convert_to_tensorrt(
            str(onnx_path), 
            str(engine_path),
            args.max_batch_size,
            args.input_height,
            args.input_width,
            args.target_width,
            args.target_height,
            fp16=args.fp16,
            workspace_size=args.workspace
        )
        
        print("\n" + "="*80)
        print("✓ Export completed successfully!")
        print(f"ONNX model: {onnx_path}")
        print(f"TensorRT engine: {engine_path}")
        print("\nUsage in code:")
        print(f"  Input shape: [3, {args.input_height}, {args.input_width}] (CHW format, max_batch_size: {args.max_batch_size})")
        print(f"  Output shape: [3, {args.target_height}, {args.target_width}] per request")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()