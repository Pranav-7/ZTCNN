import torch
from torch import nn
import ezkl
import os
import json
import asyncio
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL DEFINITION (Exact copy from notebook)
# ============================================================================

class MyModel(nn.Module):
    """
    Exact same CNN model from the notebook.
    Convolutional neural network with 2 conv layers and 2 linear layers.
    """
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)

        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x2x12x12
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (3*4*4) = 32 x 48
        x = x.flatten(start_dim=1)

        # 32 x 48 => 32x48
        x = self.d1(x)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)

        return logits


# ============================================================================
# FILE PATHS (Same as notebook)
# ============================================================================

model_path = 'network.onnx'
compiled_model_path = 'network.compiled'
pk_path = 'test.pk'
vk_path = 'test.vk'
settings_path = 'settings.json'
witness_path = 'witness.json'
data_path = 'input.json'
cal_path = 'calibration.json'
proof_path = 'test.pf'


# ============================================================================
# STEP 1: CREATE AND EXPORT MODEL
# ============================================================================

def step1_create_and_export_model():
    """Create model and export to ONNX"""
    print("\n" + "="*70)
    print("STEP 1: Creating Model and Exporting to ONNX")
    print("="*70)
    
    # Create model instance
    circuit = MyModel()
    
    # Set to evaluation mode
    circuit.eval()
    
    # Create sample input (same shape as notebook: 1x1x28x28)
    shape = [1, 28, 28]
    x = 0.1 * torch.rand(1, *shape, requires_grad=True)
    
    print(f"Model created: MyModel")
    print(f"Input shape: {x.shape}")
    
    # Export to ONNX with LEGACY exporter (compatible with EZKL)
    print("Exporting to ONNX (using legacy exporter for compatibility)...")
    
    # Force legacy ONNX exporter
    with torch.no_grad():
        torch.onnx.export(
            circuit,                        # model being run
            x,                              # model input
            model_path,                     # where to save the model
            export_params=True,             # store the trained parameter weights
            opset_version=10,               # ONNX version (10 is most compatible)
            do_constant_folding=True,       # constant folding for optimization
            input_names=['input'],          # input names
            output_names=['output'],        # output names
            dynamic_axes={
                'input': {0: 'batch_size'},    # variable length axes
                'output': {0: 'batch_size'}
            },
            # Use legacy exporter for better EZKL compatibility
            dynamo=False
        )
    
    print(f"✓ Model exported to: {model_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(model_path)
    try:
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model is valid (opset version: {onnx_model.opset_import[0].version})")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")
    
    # Create input data file (same as notebook)
    data_array = ((x).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    
    # Save to JSON
    with open(data_path, 'w') as f:
        json.dump(data, f)
    
    print(f"✓ Input data saved to: {data_path}")
    
    return shape


# ============================================================================
# STEP 2: GENERATE SETTINGS
# ============================================================================

def step2_generate_settings():
    """Generate EZKL settings (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 2: Generating EZKL Settings")
    print("="*70)
    
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "private"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # private by default
    
    print("Visibility settings:")
    print("  - Input: private")
    print("  - Output: public")
    print("  - Parameters: fixed")
    
    print("\nGenerating settings...")
    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    
    if res:
        print(f"✓ Settings generated: {settings_path}")
    else:
        raise Exception("Failed to generate settings")


# ============================================================================
# STEP 3: CALIBRATE SETTINGS
# ============================================================================

def step3_calibrate_settings(shape):
    """Calibrate settings with sample data (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 3: Calibrating Settings")
    print("="*70)
    
    # Generate calibration data (same as notebook: 20 samples)
    print("Generating calibration data (20 samples)...")
    data_array = (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()
    
    data = dict(input_data=[data_array])
    
    # Save calibration data
    with open(cal_path, 'w') as f:
        json.dump(data, f)
    
    print("Calibrating... (warnings about decomposition errors are normal)")
    result = ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
    
    if result:
        print("✓ Calibration complete")
    else:
        print("⚠ Calibration completed with warnings (this is normal)")


# ============================================================================
# STEP 4: COMPILE CIRCUIT
# ============================================================================

def step4_compile_circuit():
    """Compile the circuit (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 4: Compiling Circuit")
    print("="*70)
    
    print("Compiling circuit...")
    print("(This may take 2-5 minutes)")
    
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    
    if res:
        print(f"✓ Circuit compiled: {compiled_model_path}")
    else:
        raise Exception("Circuit compilation failed")


# ============================================================================
# STEP 5: GET SRS
# ============================================================================

async def step5_get_srs():
    """Get SRS (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 5: Getting SRS")
    print("="*70)
    
    print("Downloading SRS...")
    res = await ezkl.get_srs(settings_path)
    print("✓ SRS ready")


# ============================================================================
# STEP 6: GENERATE WITNESS
# ============================================================================

def step6_generate_witness():
    """Generate witness (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 6: Generating Witness")
    print("="*70)
    
    print("Computing witness...")
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    
    if os.path.isfile(witness_path):
        print(f"✓ Witness generated: {witness_path}")
    else:
        raise Exception("Witness generation failed")


# ============================================================================
# STEP 7: SETUP KEYS
# ============================================================================

def step7_setup_keys():
    """Setup proving and verification keys (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 7: Setting Up Keys")
    print("="*70)
    
    print("Generating proving and verification keys...")
    print("(This may take 5-10 minutes)")
    
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )
    
    if res and os.path.isfile(vk_path) and os.path.isfile(pk_path):
        print(f"✓ Verification key: {vk_path}")
        print(f"✓ Proving key: {pk_path}")
    else:
        raise Exception("Key setup failed")


# ============================================================================
# STEP 8: GENERATE PROOF
# ============================================================================

def step8_generate_proof():
    """Generate the proof (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 8: Generating Proof")
    print("="*70)
    
    print("Generating proof...")
    print("(This may take 5-10 minutes)")
    
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
    )
    
    print("\nProof generated!")
    print(f"Instances: {len(res['instances'][0])} public values")
    
    if os.path.isfile(proof_path):
        proof_size = os.path.getsize(proof_path) / 1024
        print(f"✓ Proof saved: {proof_path}")
        print(f"  Proof size: {proof_size:.2f} KB")
    else:
        raise Exception("Proof generation failed")


# ============================================================================
# STEP 9: VERIFY PROOF
# ============================================================================

def step9_verify_proof():
    """Verify the proof (same as notebook)"""
    print("\n" + "="*70)
    print("STEP 9: Verifying Proof")
    print("="*70)
    
    print("Verifying proof...")
    
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )
    
    if res:
        print("\n" + "="*70)
        print("✓✓✓ VERIFIED! ✓✓✓")
        print("="*70)
        print("\nSuccess! The proof is valid.")
        print("This proves you ran the CNN model on private input data!")
    else:
        raise Exception("Verification failed")


# ============================================================================
# SUMMARY
# ============================================================================

def show_summary():
    """Show summary of all generated files"""
    print("\n" + "="*70)
    print("SUMMARY - Generated Files")
    print("="*70)
    
    files = [
        ("ONNX Model", model_path),
        ("Compiled Model", compiled_model_path),
        ("Settings", settings_path),
        ("Calibration Data", cal_path),
        ("Input Data", data_path),
        ("Witness", witness_path),
        ("Proving Key", pk_path),
        ("Verification Key", vk_path),
        ("Proof", proof_path),
    ]
    
    print("\nAll files in current directory:\n")
    for name, path in files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            exists = "✓"
        else:
            size = 0
            exists = "✗"
        print(f"  {exists} {name:20s}: {path:25s} ({size:8.1f} KB)")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution - runs all steps in sequence"""
    print("\n" + "="*70)
    print("EZKL SIMPLE CNN DEMO")
    print("Direct port from Jupyter notebook")
    print("="*70)
    print("\nThis will run through all 9 steps:")
    print("  1. Create and export CNN model to ONNX")
    print("  2. Generate EZKL settings")
    print("  3. Calibrate settings")
    print("  4. Compile circuit")
    print("  5. Get SRS")
    print("  6. Generate witness")
    print("  7. Setup proving/verification keys")
    print("  8. Generate zero-knowledge proof")
    print("  9. Verify the proof")
    print("\nEstimated time: 15-25 minutes")
    print("="*70)
    
    input("\nPress Enter to start...")
    
    try:
        # Step 1: Create and export model
        shape = step1_create_and_export_model()
        
        # Step 2: Generate settings
        step2_generate_settings()
        
        # Step 3: Calibrate
        step3_calibrate_settings(shape)
        
        # Step 4: Compile circuit
        step4_compile_circuit()
        
        # Step 5: Get SRS (async)
        asyncio.run(step5_get_srs())
        
        # Step 6: Generate witness
        step6_generate_witness()
        
        # Step 7: Setup keys
        step7_setup_keys()
        
        # Step 8: Generate proof
        step8_generate_proof()
        
        # Step 9: Verify proof
        step9_verify_proof()
        
        # Show summary
        show_summary()
        
        print("\n" + "="*70)
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you installed: pip install torch==2.0.1 ezkl onnx")
        print("2. Check Python version: python --version (need 3.8-3.11)")
        print("3. Ensure you have ~500MB free disk space")
        print("4. Check virtual environment is activated")
        print("5. If ONNX export fails, try: pip uninstall torch && pip install torch==2.0.1")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()