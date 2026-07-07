"""Human-simulator server: a Qwen3-VL model that role-plays the USER during an
Interactive-RoboMME rollout.

It is NOT part of the robot. The robot is the frozen pi0.5. This server is the
evaluation-side "human": every time the sim client calls /interject, it receives
the current camera frame + the task + the GROUND-TRUTH goal (what the user wants),
and decides — like a real person watching — whether to say something right now,
and if so, what. This replaces the hard-coded "utter a fixed line at step 10".

Key design point: the model is TOLD the ground truth (the user knows what they
want; e.g. which blue cube). Its job is dynamic timing + natural wording, NOT
inferring the answer from pixels (a preference isn't visible in the image).

Run (from robomme_policy_learning, human-sim venv):
  CUDA_VISIBLE_DEVICES=0 <humansim_venv python> examples/robomme/human_sim_server.py \
    --port 8001 --model Qwen/Qwen3-VL-4B-Instruct
"""

import argparse
import base64
import io
import json

import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)
STATE = {"model": None, "processor": None, "model_name": None}


SYSTEM_PROMPT = (
    "You are a person giving a tabletop robot short spoken instructions. "
    "You KNOW what you want (it is given to you as GOAL). Watch the robot's current "
    "camera view and decide, like a real person would, whether to speak RIGHT NOW.\n"
    "- Speak only when it helps: the task is ambiguous/underspecified, or the robot is "
    "about to do the wrong thing, or it seems stuck.\n"
    "- If everything looks fine and no help is needed yet, say nothing.\n"
    "- When you do speak, use ONE short, natural, spoken sentence — the way a person "
    "actually talks to a robot. Do not read out coordinates or internal state.\n"
    "Reply as strict JSON: {\"speak\": true/false, \"utterance\": \"...\"}. "
    "If speak is false, utterance is \"\"."
)


def load_model(model_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[human-sim] loading {model_name} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    STATE.update(model=model, processor=processor, model_name=model_name)
    print("[human-sim] model ready", flush=True)


def _decode_image(payload) -> Image.Image:
    if isinstance(payload, str):  # base64 PNG/JPEG
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    arr = np.asarray(payload, dtype=np.uint8)  # raw HxWx3 list
    return Image.fromarray(arr).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ready": STATE["model"] is not None, "model": STATE["model_name"]})


@app.route("/interject", methods=["POST"])
def interject():
    """Body: {image, task, goal, step, already_said:[...]}. Returns {speak, utterance}."""
    data = request.get_json(force=True)
    img = _decode_image(data["image"])
    task = data.get("task", "")
    goal = data.get("goal", "")
    step = data.get("step", 0)
    already = data.get("already_said", [])

    user_text = (
        f"TASK: {task}\n"
        f"GOAL (what you, the user, want): {goal}\n"
        f"Robot step: {step}. "
        + (f"You already said: {already}. Don't repeat yourself. " if already else "")
        + "Look at the current view. Speak now?"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]},
    ]
    processor = STATE["processor"]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(STATE["model"].device)

    with torch.no_grad():
        out = STATE["model"].generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen, skip_special_tokens=True).strip()

    speak, utterance = _parse(text)
    return jsonify({"speak": speak, "utterance": utterance, "raw": text})


def _parse(text: str):
    """Parse the model's JSON reply, tolerating markdown fences / stray prose."""
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json").strip() if s.count("```") >= 2 else s
    try:
        start, end = s.index("{"), s.rindex("}") + 1
        obj = json.loads(s[start:end])
        speak = bool(obj.get("speak", False))
        utt = str(obj.get("utterance", "")).strip()
        return (speak and bool(utt)), (utt if speak else "")
    except Exception:
        return False, ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()
    load_model(args.model)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
