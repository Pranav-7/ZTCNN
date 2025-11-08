import os
import json
import asyncio
import warnings
import traceback
import sys
import subprocess
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from torch import nn
import onnx

try:
    import ezkl
except Exception:
    ezkl = None



warnings.filterwarnings("ignore")

app = FastAPI(title="EZKL CNN Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# File paths (same as the notebook)
# ------------------------------
MODEL_PATH = "network.onnx"
COMPILED_MODEL_PATH = "network.compiled"
PK_PATH = "test.pk"
VK_PATH = "test.vk"
SETTINGS_PATH = "settings.json"
WITNESS_PATH = "witness.json"
DATA_PATH = "input.json"
CAL_PATH = "calibration.json"
PROOF_PATH = "test.pf"

# ------------------------------
# Minimal in-memory logger to collect step messages
# ------------------------------
class StepLogger:
    def __init__(self):
        self.lines: List[str] = []

    def log(self, *args, **kwargs):
        line = " ".join(map(str, args))
        self.lines.append(line)
        print(line, **kwargs)

    def get(self) -> List[str]:
        return self.lines

    def clear(self):
        self.lines.clear()

logger = StepLogger()

# ------------------------------
# Model definition (exact as notebook)
# ------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)

        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.d1(x)
        x = self.relu(x)
        logits = self.d2(x)
        return logits

# ------------------------------
# Helper: file status listing
# ------------------------------
def file_status(path: str) -> Dict[str, Any]:
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    return {"path": path, "exists": exists, "size_kb": size_kb}

# ------------------------------
# Steps implementations (adapted to return logs/results)
# Each function appends to logger and returns a dict result.
# ------------------------------
def step1_create_and_export_model() -> Dict[str, Any]:
    """
    Run the export code in a fresh Python subprocess by creating a small
    temporary runner script that imports cnn_code.step1_create_and_export_model().
    This avoids -c quoting issues.
    """
    logger.log("=" * 60)
    logger.log("STEP 1: Creating Model and Exporting to ONNX (subprocess via temp file)")
    logger.log("=" * 60)

    cwd = Path.cwd()

    # Build python runner source
    runner_src = r"""
import json, traceback, sys
from cnn_code import step1_create_and_export_model

try:
    shape = step1_create_and_export_model()
    # print a machine-parsable marker so the parent process can parse shape if desired
    print("EXPORT_OK_SHAPE=" + json.dumps(shape))
except Exception as e:
    traceback.print_exc()
    # ensure non-zero exit
    sys.exit(1)
"""

    # Create temporary file in cwd (so relative imports and files are in same folder)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(cwd), delete=False) as tf:
        tf_path = Path(tf.name)
        tf.write(runner_src)
        tf.flush()

    cmd = [sys.executable, str(tf_path)]

    logger.log("Running subprocess:", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        logger.log("Subprocess returned non-zero exit code.")
        logger.log("stdout:\n" + (e.stdout or ""))
        logger.log("stderr:\n" + (e.stderr or ""))
        # remove temp file
        try:
            tf_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Subprocess export failed: {e.returncode}\nSee server logs for stdout/stderr")
    except subprocess.TimeoutExpired as e:
        logger.log("Subprocess timed out.")
        logger.log("stdout:\n" + (e.stdout or ""))
        logger.log("stderr:\n" + (e.stderr or ""))
        try:
            tf_path.unlink()
        except Exception:
            pass
        raise RuntimeError("Subprocess export timed out (timeout=600s)")
    except Exception as e:
        logger.log("Subprocess execution error:", repr(e))
        try:
            tf_path.unlink()
        except Exception:
            pass
        raise

    # remove temp file now that it's done
    try:
        tf_path.unlink()
    except Exception:
        pass

    # Log outputs
    if proc.stdout:
        logger.log("Subprocess stdout:\n" + proc.stdout)
    if proc.stderr:
        logger.log("Subprocess stderr:\n" + proc.stderr)

    # parse exported shape if present
    exported_shape = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("EXPORT_OK_SHAPE="):
            try:
                exported_shape = json.loads(line.split("=", 1)[1])
            except Exception:
                exported_shape = None

    # file statuses
    model_info = file_status(MODEL_PATH)
    data_info = file_status(DATA_PATH)
    if not model_info["exists"]:
        logger.log("Export subprocess completed but ONNX file not found:", MODEL_PATH)
        logger.log("Subprocess stdout:\n" + (proc.stdout or ""))
        logger.log("Subprocess stderr:\n" + (proc.stderr or ""))
        raise RuntimeError("ONNX export subprocess finished but network.onnx not created. See logs above.")

    # try read opset
    try:
        import onnx as _onnx
        m = _onnx.load(MODEL_PATH)
        opset_ver = m.opset_import[0].version if m.opset_import else None
        logger.log(f"Detected opset in exported ONNX: {opset_ver}")
        model_info["opset"] = opset_ver
    except Exception as e:
        logger.log("Failed to read opset from exported ONNX:", repr(e))

    logger.log(f"✓ Model exported by subprocess to: {MODEL_PATH}")
    logger.log(f"✓ Input data saved to: {DATA_PATH}")

    return {"ok": True, "shape": exported_shape or [1, 28, 28], "files": [model_info, data_info]}

def step2_generate_settings() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 2: Generating EZKL Settings")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "private"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"
    logger.log("Generating settings...")
    res = ezkl.gen_settings(MODEL_PATH, SETTINGS_PATH, py_run_args=py_run_args)
    if not res:
        raise RuntimeError("Failed to generate settings via ezkl.gen_settings")
    logger.log(f"✓ Settings generated: {SETTINGS_PATH}")
    return {"ok": True, "files": [file_status(SETTINGS_PATH)]}

def step3_calibrate_settings(shape: Optional[List[int]] = None) -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 3: Calibrating Settings")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    if shape is None:
        # default to 1x28x28
        shape = [1, 28, 28]
    logger.log("Generating calibration data (20 samples)...")
    data_array = (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    with open(CAL_PATH, "w") as f:
        json.dump(data, f)
    logger.log("Calibrating... (may show warnings — that's normal)")
    result = ezkl.calibrate_settings(CAL_PATH, MODEL_PATH, SETTINGS_PATH, "resources")
    # result can be truthy/falsey; return it
    logger.log("Calibration result:", result)
    return {"ok": True, "files": [file_status(CAL_PATH)], "result": str(result)}

def step4_compile_circuit() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 4: Compiling Circuit")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Compiling circuit... (may take minutes)")
    res = ezkl.compile_circuit(MODEL_PATH, COMPILED_MODEL_PATH, SETTINGS_PATH)
    if not res:
        raise RuntimeError("ezkl.compile_circuit returned a falsy result")
    logger.log(f"✓ Circuit compiled: {COMPILED_MODEL_PATH}")
    return {"ok": True, "files": [file_status(COMPILED_MODEL_PATH)]}

async def step5_get_srs() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 5: Getting SRS")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Downloading SRS (may require network and time)...")
    res = await ezkl.get_srs(SETTINGS_PATH)
    logger.log("✓ SRS ready")
    return {"ok": True, "srs": str(res)}

def step6_generate_witness() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 6: Generating Witness")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Computing witness...")
    res = ezkl.gen_witness(DATA_PATH, COMPILED_MODEL_PATH, WITNESS_PATH)
    if not os.path.isfile(WITNESS_PATH):
        raise RuntimeError("Witness generation failed; witness file not found.")
    logger.log(f"✓ Witness generated: {WITNESS_PATH}")
    return {"ok": True, "files": [file_status(WITNESS_PATH)], "result": str(res)}

def step7_setup_keys() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 7: Setting Up Keys")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Generating proving and verification keys (may take minutes)...")
    res = ezkl.setup(COMPILED_MODEL_PATH, VK_PATH, PK_PATH)
    if not (os.path.isfile(VK_PATH) and os.path.isfile(PK_PATH)):
        raise RuntimeError("Key setup failed; keys not found.")
    logger.log(f"✓ Verification key: {VK_PATH}")
    logger.log(f"✓ Proving key: {PK_PATH}")
    return {"ok": True, "files": [file_status(VK_PATH), file_status(PK_PATH)], "result": str(res)}

def step8_generate_proof() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 8: Generating Proof")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Generating proof (may take minutes)...")
    res = ezkl.prove(WITNESS_PATH, COMPILED_MODEL_PATH, PK_PATH, PROOF_PATH)
    if not os.path.isfile(PROOF_PATH):
        raise RuntimeError("Proof generation failed; proof file not found.")
    proof_size_kb = os.path.getsize(PROOF_PATH) / 1024
    instances_count = len(res["instances"][0]) if ("instances" in res and res["instances"]) else None
    logger.log(f"✓ Proof saved: {PROOF_PATH} ({proof_size_kb:.2f} KB)")
    return {"ok": True, "files": [file_status(PROOF_PATH)], "instances": instances_count, "res": str(res)}

def step9_verify_proof() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("STEP 9: Verifying Proof")
    logger.log("=" * 60)
    if ezkl is None:
        raise RuntimeError("ezkl is not installed or failed to import.")
    logger.log("Verifying proof...")
    res = ezkl.verify(PROOF_PATH, SETTINGS_PATH, VK_PATH)
    if not res:
        raise RuntimeError("Verification failed.")
    logger.log("✓✓✓ VERIFIED! ✓✓✓")
    return {"ok": True, "verified": True}

def show_summary() -> Dict[str, Any]:
    logger.log("=" * 60)
    logger.log("SUMMARY - Generated Files")
    logger.log("=" * 60)
    files = [
        ("ONNX Model", MODEL_PATH),
        ("Compiled Model", COMPILED_MODEL_PATH),
        ("Settings", SETTINGS_PATH),
        ("Calibration Data", CAL_PATH),
        ("Input Data", DATA_PATH),
        ("Witness", WITNESS_PATH),
        ("Proving Key", PK_PATH),
        ("Verification Key", VK_PATH),
        ("Proof", PROOF_PATH),
    ]
    out = []
    for name, path in files:
        info = file_status(path)
        info["label"] = name
        out.append(info)
        logger.log(("[✓]" if info["exists"] else "[ ]"), f"{name:20s}: {path:25s} ({info['size_kb']:.1f} KB)")
    return {"ok": True, "files": out}

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "ezkl_installed": ezkl is not None}

@app.get("/status")
async def status():
    # show the set of file statuses
    return {
        "ezkl_installed": ezkl is not None,
        "files": {
            "model": file_status(MODEL_PATH),
            "compiled": file_status(COMPILED_MODEL_PATH),
            "settings": file_status(SETTINGS_PATH),
            "witness": file_status(WITNESS_PATH),
            "proof": file_status(PROOF_PATH),
            "pk": file_status(PK_PATH),
            "vk": file_status(VK_PATH),
            "input": file_status(DATA_PATH),
            "calibration": file_status(CAL_PATH),
        }
    }

@app.post("/clear_logs")
async def clear_logs():
    logger.clear()
    return {"ok": True}

@app.get("/logs")
async def get_logs():
    return {"lines": logger.get()}

@app.post("/step/{step_id}")
async def run_step(step_id: int):
    """
    Run a single step (1-9). Returns logs and result.
    Step 5 is async and will be awaited.
    """
    logger.clear()
    try:
        if step_id == 1:
            result = step1_create_and_export_model()
        elif step_id == 2:
            result = step2_generate_settings()
        elif step_id == 3:
            # attempt to infer shape from existing input if present
            shape = None
            if os.path.exists(DATA_PATH):
                try:
                    with open(DATA_PATH, "r") as f:
                        d = json.load(f)
                        if "input_data" in d and len(d["input_data"]) > 0:
                            # try to deduce length -> assume [1,28,28]
                            shape = [1, 28, 28]
                except Exception:
                    shape = [1, 28, 28]
            result = step3_calibrate_settings(shape)
        elif step_id == 4:
            result = step4_compile_circuit()
        elif step_id == 5:
            # async
            result = await step5_get_srs()
        elif step_id == 6:
            result = step6_generate_witness()
        elif step_id == 7:
            result = step7_setup_keys()
        elif step_id == 8:
            result = step8_generate_proof()
        elif step_id == 9:
            result = step9_verify_proof()
        else:
            raise HTTPException(status_code=400, detail="Unknown step_id. Valid: 1..9")
        logs = logger.get()
        return {"ok": True, "step": step_id, "result": result, "logs": logs}
    except Exception as e:
        tb = traceback.format_exc()
        logs = logger.get()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb, "logs": logs})

@app.post("/run_all")
async def run_all():
    """
    Run all steps sequentially: 1..9
    This endpoint runs synchronously and returns when finished (or when an error occurs).
    """
    logger.clear()
    results = {}
    try:
        # Step 1
        results["1"] = step1_create_and_export_model()
        # Step 2
        results["2"] = step2_generate_settings()
        # Step 3 (pass shape from step1)
        results["3"] = step3_calibrate_settings(results["1"].get("shape"))
        # Step 4
        results["4"] = step4_compile_circuit()
        # Step 5 (async)
        results["5"] = await step5_get_srs()
        # Step 6
        results["6"] = step6_generate_witness()
        # Step 7
        results["7"] = step7_setup_keys()
        # Step 8
        results["8"] = step8_generate_proof()
        # Step 9
        results["9"] = step9_verify_proof()

        # summary
        results["summary"] = show_summary()
        logs = logger.get()
        return {"ok": True, "results": results, "logs": logs}
    except Exception as e:
        tb = traceback.format_exc()
        logs = logger.get()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb, "logs": logs})

@app.get("/files")
async def list_files():
    return show_summary()
