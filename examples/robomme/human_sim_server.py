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
    "You are the USER standing next to a tabletop robot arm, watching it try to do a "
    "task. You KNOW exactly what you want — it is given to you as GOAL — but the robot "
    "was only given a vague instruction, so it may not know. Each time you are shown the "
    "robot's current camera view, decide like a real person whether to speak RIGHT NOW.\n"
    "\n"
    "SPEAK when the view shows one of these situations:\n"
    "1. WRONG TARGET: the arm is reaching toward, hovering over, or about to grasp the "
    "WRONG object (not your GOAL). Say a short correction, e.g. \"no, not that one — the "
    "red one\".\n"
    "2. WRONG METHOD: the robot is about to do the action the wrong way (e.g. grasping "
    "when you wanted it pushed). Say e.g. \"don't grab it, push it\".\n"
    "3. ABOUT TO VIOLATE a constraint/preference you hold (e.g. it is reaching for a cube "
    "you never want touched). Say e.g. \"leave the green one, don't touch it\".\n"
    "4. STUCK / WANDERING: the arm hovers, drifts, or has made no progress for a while, "
    "clearly unsure. Give the missing information, e.g. \"the one on the left\".\n"
    "\n"
    "STAY SILENT when: the arm is still moving into position and hasn't committed to "
    "anything wrong yet; or it is already doing the right thing; or you have nothing new "
    "to add. When unsure, prefer to stay silent — a real person doesn't narrate every "
    "moment, they speak up mainly when the robot is going wrong or clearly stuck.\n"
    "\n"
    "When you DO speak: ONE short, natural, spoken sentence, the way a person actually "
    "talks to a robot. Refer to objects the way a person would (colour, 'the left one', "
    "'that one'). Never read out coordinates or internal state. Do not repeat something "
    "you already said.\n"
    "\n"
    "Reply as strict JSON only: {\"speak\": true|false, \"utterance\": \"...\"}. "
    "If speak is false, utterance must be \"\"."
)


def load_model(model_name: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"[human-sim] loading {model_name} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
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

    hint = data.get("robot_state_hint", "")  # optional coarse cue from the harness
    user_text = (
        f"TASK the robot was given (vague): {task}\n"
        f"GOAL — what you actually want: {goal}\n"
        f"Robot step: {step}.\n"
        + (f"Observed: {hint}\n" if hint else "")
        + (f"You have already said: {already} — do NOT repeat these.\n" if already else "")
        + "Look carefully at what the arm is doing and which object it is reaching "
        "toward. Is it going wrong or stuck relative to your GOAL? Decide whether to "
        "speak now, and reply with the JSON."
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
