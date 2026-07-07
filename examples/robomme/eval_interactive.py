"""Interactive-RoboMME evaluation harness — unified 4 settings per task on a frozen
pi05_baseline policy (no training, original scorers untouched).

Four settings, identical names across all four tasks:
  baseline       vague instruction, visual cue OFF -> the info the robot needs is
                 missing (the real scenario: a user gives a vague command).
  utterance      same vague start, but at step 10 the correct answer is injected as
                 one line of text (oracle upper bound: "if told, can pi0.5 use it?").
  original       the benchmark's own visual cue is ON (highlight / lift / demo video /
                 full instruction) — reference for "with the cue, how well does it do".
  utterance_vlm  vague start, cue OFF; a Qwen3-VL "human" (human_sim_server) watches
                 every N steps and decides itself whether/when/what to say.

Per-task specifics live in TASK_CONFIG. PickHighlight runs only on same-colour-ambiguity
episodes (task 1). MoveCube's `original` uses history (so the policy sees the demo
video); its other settings run no-history (pure prior).
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import EpisodeState, RolloutRecorder, pack_buffer
from env_runner import EnvRunner


CONDITIONS = ["baseline", "utterance", "original", "utterance_vlm"]
TASKS = ["PickHighlight", "ButtonUnmask", "MoveCube", "BinFill"]


# ------------------------------------------------------------------ ground truth

def _pos(obj):
    p = obj.pose.p if hasattr(obj, "pose") else obj.p
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p).reshape(-1)[:3]


def target_color(u):
    return u.target_cube_colors[0]


def target_color_unique(u) -> bool:
    return all(u.all_cube_colors.count(c) == 1 for c in u.target_cube_colors)


def bin_position_word(u) -> str:
    """ButtonUnmask: target is always bin_0; front camera image-right = +y."""
    ys = [float(_pos(b)[1]) for b in u.spawned_bins]
    rank = sorted(range(len(ys)), key=lambda i: ys[i]).index(0)
    return ("left", "middle", "right")[rank] if len(ys) == 3 else f"{rank + 1}th from the left"


def pick_position_word(u) -> str:
    """PickHighlight: rank of the target among same-colour cubes, left->right (+y)."""
    tc = target_color(u)
    same_ys = sorted(float(_pos(u.all_cubes[i])[1])
                     for i in range(len(u.all_cubes)) if u.all_cube_colors[i] == tc)
    ty = float(_pos(u.target_cubes[0])[1])
    rank = min(range(len(same_ys)), key=lambda k: abs(same_ys[k] - ty))
    n = len(same_ys)
    if n == 2:
        return "left" if rank == 0 else "right"
    if n == 3:
        return ("left", "middle", "right")[rank]
    return f"{rank + 1}-th from the left"


MOVECUBE_WAY_PROMPT = {
    "peg_push": "pick up the peg and use it to push the cube onto the target - do not grasp the cube",
    "gripper_push": "push the cube onto the target with your gripper - do not grasp the cube",
    "grasp_putdown": "pick up the cube and place it down on the target",
}
MOVECUBE_WAY_GOAL = {
    "peg_push": "push the cube onto the target using the peg (do NOT grasp the cube)",
    "gripper_push": "push the cube onto the target with the gripper (do NOT grasp the cube)",
    "grasp_putdown": "grasp the cube and place it down on the target",
}


def binfill_pref(u):
    return {"red": getattr(u, "red_cubes_target_number", 0),
            "blue": getattr(u, "blue_cubes_target_number", 0),
            "green": getattr(u, "green_cubes_target_number", 0)}


def binfill_goal(u) -> str:
    parts = [f"{n} {c}" for c, n in binfill_pref(u).items() if n > 0]
    return "put " + ", ".join(parts) + " cube(s) into the bin"


# ---- per-task behaviour: vague prompt / informed answer / vlm goal / scene facts ----

def _pick_vague(u, tg):
    return f"first press the button, then pick up the {target_color(u)} cube"  # 2 same-colour


def _pick_informed(u, tg):
    return f"first press the button, then pick up the {pick_position_word(u)} {target_color(u)} cube"


def _pick_goal(u):
    return f"the {pick_position_word(u)} {target_color(u)} cube"


def _pick_scene(u):
    from collections import Counter
    cs = list(u.all_cube_colors); cnt = Counter(cs)
    dup = [c for c, n in cnt.items() if n > 1]
    s = f"On the table there are {len(cs)} cubes, colours: {cs}."
    if dup:
        s += " Note: " + ", ".join(f"{cnt[c]} {c} cubes" for c in dup) + " — naming only that colour is ambiguous."
    return s


def _btn_informed(u, tg):
    return f"first press the button, then pick up the {bin_position_word(u)} container"


def _btn_goal(u):
    return f"the {bin_position_word(u)} container (it hides the {u.color_names[0]} cube)"


def _btn_scene(u):
    n = len(u.spawned_bins)
    return (f"On the table there are {n} identical closed containers in a row; you cannot "
            f"see which cube is under which — but you know your target is the "
            f"{bin_position_word(u)} one.")


def _move_informed(u, tg):
    return MOVECUBE_WAY_PROMPT[u.way]


def _move_goal(u):
    return MOVECUBE_WAY_GOAL[u.way]


def _move_scene(u):
    return ("A single cube and a target marker are on the table. The robot can move the "
            "cube in different ways (push with a peg, push with the gripper, or grasp and "
            "place). Watch whether it is grasping when it should push, or vice versa.")


def _bin_informed(u, tg):
    return tg  # the benchmark's real full instruction (encodes the preference)


def _bin_scene(u):
    from collections import Counter
    return (f"Coloured cubes and one bin are on the table. You have a preference for what "
            f"goes in: {binfill_goal(u)}. Watch whether the robot puts in the wrong colour "
            f"or the wrong number.")


TASK_CONFIG = {
    "PickHighlight": dict(cue_off="robomme_disable_highlight", ambiguous_only=True,
                          original_history=False, vague=_pick_vague, informed=_pick_informed,
                          goal=_pick_goal, scene=_pick_scene),
    "ButtonUnmask": dict(cue_off="robomme_disable_lift", ambiguous_only=False,
                         original_history=False, vague=lambda u, tg: tg, informed=_btn_informed,
                         goal=_btn_goal, scene=_btn_scene),
    "MoveCube": dict(cue_off=None, ambiguous_only=False, original_history=True,
                     vague=lambda u, tg: "move the cube onto the target", informed=_move_informed,
                     goal=_move_goal, scene=_move_scene),
    "BinFill": dict(cue_off=None, ambiguous_only=False, original_history=False,
                    vague=lambda u, tg: "put the cubes I like into the bin, then press the button to stop",
                    informed=_bin_informed, goal=binfill_goal, scene=_bin_scene),
}


# ------------------------------------------------------------------- VLM client

def call_vlm_human(args, img, task, goal, step, already, hint=""):
    import cv2, base64, requests
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    payload = {"image": base64.b64encode(buf).decode(), "task": task, "goal": goal,
               "step": int(step), "already_said": already, "robot_state_hint": hint}
    r = requests.post(f"http://{args.vlm_host}:{args.vlm_port}/interject", json=payload, timeout=60)
    return r.json()


def approaching_wrong_hint(u, proximity=0.12):
    """Difficulty-2 helper: coarse action cue a person would see — is the gripper
    heading for a non-target object? Only meaningful for PickHighlight."""
    try:
        tcp = _pos(u.agent.tcp_pose)
        nts = [c for c in u.all_cubes if c not in u.target_cubes]
        if nts and min(float(np.linalg.norm(tcp - _pos(c))) for c in nts) < proximity:
            return "The robot arm is currently reaching toward one of the cubes that is NOT your target."
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------- Args

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    task: str = "PickHighlight"
    condition: str = "baseline"
    utterance_step: int = 10
    num_episodes: int = 10
    difficulty: str = "easy"
    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation_interactive"
    model_seed: int = 7
    use_history: bool = False      # forced True internally for MoveCube/original
    vlm_host: str = "0.0.0.0"
    vlm_port: int = 8001
    vlm_every: int = 20


class InteractiveEnvRunner(EnvRunner):
    def __init__(self, env_id, video_save_dir, max_steps=1300, extra_env_kwargs=None):
        super().__init__(env_id, video_save_dir, max_steps=max_steps)
        self.extra_env_kwargs = dict(extra_env_kwargs or {})

    def make_env(self, episode_id: int) -> None:
        self.env = self.env_builder.make_env_for_episode(
            episode_id, extra_env_kwargs=self.extra_env_kwargs)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty

    def episodes_with_difficulty(self, difficulty: str):
        return [ep for ep in range(self.env_builder.get_episode_num())
                if self.env_builder.resolve_episode(ep)[1] == difficulty]


# ------------------------------------------------------ per-episode controller

class Controller:
    """Owns the prompt at every chunk boundary for the 4 unified settings."""

    def __init__(self, args, task_goal, env):
        self.args = args
        self.c = args.condition
        self.task = args.task
        self.cfg = TASK_CONFIG[args.task]
        self.u = env.unwrapped
        self.task_goal = task_goal
        self.events = []
        self.n_interject = 0
        self._pending = None
        self.vague = self.cfg["vague"](self.u, task_goal)
        self.informed = self.cfg["informed"](self.u, task_goal)
        self.vlm_goal_str = self.cfg["goal"](self.u)

    def initial_prompt(self) -> str:
        if self.c == "original":
            return self.task_goal          # benchmark's own instruction, cue ON
        return self.vague                  # baseline / utterance / utterance_vlm start vague

    def on_step(self, step: int):
        # utterance: inject the correct answer once at utterance_step
        if self.c == "utterance" and step >= self.args.utterance_step and self._pending is None:
            self._pending = self.informed
            self.events.append({"step": int(step), "kind": "tell", "text": self.informed})
            print(f"[scripted user] step {step} [tell]: {self.informed!r}", flush=True)

    def vlm_said(self):
        return [e["text"] for e in self.events if e["kind"] == "vlm"]

    def add_vlm(self, step, text):
        self._pending = text
        self.n_interject += 1
        self.events.append({"step": int(step), "kind": "vlm", "text": text})
        print(f"[vlm human] step {step}: {text!r}", flush=True)

    def current_prompt(self) -> str:
        return self._pending if self._pending is not None else self.initial_prompt()

    def episode_meta(self):
        meta = {"events": self.events, "n_interject": self.n_interject}
        if self.task == "MoveCube":
            meta["sampled_way"] = self.u.way
        return meta


def episode_skip_reason(args, u) -> str:
    if TASK_CONFIG[args.task]["ambiguous_only"]:
        return "" if not target_color_unique(u) else "skipped_unambiguous"
    return ""


# ------------------------------------------------------------------- main loop

def run_episode(args, env_runner, video_save_dir):
    use_history = args.use_history or (args.task == "MoveCube" and args.condition == "original")
    client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    epstate = EpisodeState()
    pre_traj = env_runner.get_init_obs()
    task_goal = pre_traj["task_goal"]
    ctrl = Controller(args, task_goal, env_runner.env)

    recorder = RolloutRecorder(video_save_dir, ctrl.initial_prompt(), fps=30)
    print(f"task_goal: {task_goal}  |  init prompt: {ctrl.initial_prompt()!r}  |  "
          f"history={use_history}", flush=True)

    epstate.image_buffer.extend(pre_traj["images"])
    epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
    epstate.state_buffer.extend(pre_traj["states"])
    for i in range(len(pre_traj["images"])):
        recorder.record(image=pre_traj["images"][i].copy(),
                        wrist_image=pre_traj["wrist_images"][i].copy(),
                        state=pre_traj["states"][i].copy())
    epstate.exec_start_idx = len(epstate.image_buffer) - 1

    img, wrist_img, robot_state = epstate.get_current_obs()
    success_flag = "unknown"

    while True:
        if not epstate.action_plan:
            prompt = ctrl.current_prompt()
            if prompt != ctrl.initial_prompt() or ctrl.events:
                recorder.task_goal = "[HUMAN] " + prompt
            if use_history:
                r = client.add_buffer(pack_buffer(epstate.image_buffer, epstate.state_buffer,
                                                  epstate.exec_start_idx))
                while not r.get("add_buffer_finished", False):
                    time.sleep(0.1)
            element = {"observation/image": img, "observation/wrist_image": wrist_img,
                       "observation/state": robot_state, "prompt": prompt}
            action_chunk = client.infer(element)["actions"]
            epstate.action_plan.extend(action_chunk[: args.obs_horizon])
            epstate.clear_buffers()

        action = epstate.action_plan.popleft()
        obs, stop_flag, success_flag = env_runner.step(action)
        epstate.count += 1
        ctrl.on_step(epstate.count)

        if args.condition == "utterance_vlm" and epstate.count % args.vlm_every == 0:
            hint = ctrl.cfg["scene"](ctrl.u)
            if args.task == "PickHighlight":
                aw = approaching_wrong_hint(ctrl.u)
                if aw:
                    hint += " " + aw
            try:
                resp = call_vlm_human(args, img, task_goal, ctrl.vlm_goal_str,
                                      epstate.count, ctrl.vlm_said(), hint=hint)
            except Exception as exc:
                print(f"[vlm] call failed at step {epstate.count}: {exc}", flush=True)
                resp = {"speak": False}
            if resp.get("speak") and resp.get("utterance"):
                ctrl.add_vlm(epstate.count, resp["utterance"])

        if epstate.count > args.max_steps:
            success_flag = "timeout"
            break
        img, wrist_img, robot_state = obs
        epstate.add_observation(img, wrist_img, robot_state)
        recorder.record(image=img.copy(), wrist_image=wrist_img.copy(),
                        state=robot_state.copy(), action=action.copy())
        if stop_flag:
            break

    if success_flag != "unknown":
        _pad_frames(recorder)
        recorder.save_video(f"{env_runner.env_id}_ep{env_runner.episode_id}_"
                            f"{args.condition}_{success_flag}_{env_runner.difficulty}.mp4")
    return success_flag, ctrl.episode_meta()


def _pad_frames(recorder):
    frames = recorder.total_images
    if not frames:
        return
    h = max(f.shape[0] for f in frames); w = max(f.shape[1] for f in frames)
    for i, f in enumerate(frames):
        if f.shape[0] != h or f.shape[1] != w:
            canvas = np.zeros((h, w, f.shape[2]), dtype=f.dtype)
            canvas[: f.shape[0], : f.shape[1]] = f
            frames[i] = canvas


def evaluate(args: Args):
    assert args.task in TASK_CONFIG, f"task must be one of {TASKS}"
    assert args.condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    save_dir = Path(args.save_dir) / args.task / args.condition / f"seed{args.model_seed}"
    video_save_dir = save_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    progress_path = save_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        print(f"[interactive] resuming: {len(progress.get('episodes', {}))} recorded", flush=True)
    else:
        progress = {"task": args.task, "condition": args.condition, "episodes": {}, "meta": {}}

    extra = {}
    cue_off = TASK_CONFIG[args.task]["cue_off"]
    if cue_off and args.condition != "original":   # baseline/utterance/vlm hide the cue
        extra[cue_off] = True

    env_runner = InteractiveEnvRunner(args.task, video_save_dir,
                                      max_steps=args.max_steps, extra_env_kwargs=extra)
    candidates = env_runner.episodes_with_difficulty(args.difficulty)
    print(f"[interactive] {args.task}/{args.condition} candidates={candidates}", flush=True)

    def n_valid():
        return sum(1 for v in progress["episodes"].values() if not str(v).startswith("skipped_"))

    for ep in candidates:
        if n_valid() >= args.num_episodes:
            break
        if str(ep) in progress["episodes"]:
            continue
        env_runner.make_env(ep)
        reason = episode_skip_reason(args, env_runner.env.unwrapped)
        if reason:
            progress["episodes"][str(ep)] = reason
            env_runner.close_env()
            progress_path.write_text(json.dumps(progress, indent=2))
            continue
        print(f"\n[interactive] {args.task} ep{ep} ({args.condition})", flush=True)
        flag, meta = run_episode(args, env_runner, video_save_dir)
        progress["episodes"][str(ep)] = flag
        progress["meta"][str(ep)] = meta
        env_runner.close_env()
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"[interactive] ep{ep} -> {flag}", flush=True)

    done = {k: v for k, v in progress["episodes"].items() if not str(v).startswith("skipped_")}
    n = len(done); succ = sum(1 for v in done.values() if v == "success")
    tout = sum(1 for v in done.values() if v == "timeout")
    interj = sum(progress["meta"].get(k, {}).get("n_interject", 0) for k in done)
    summary = {"task": args.task, "condition": args.condition, "episodes": n, "success": succ,
               "success_rate": succ / n if n else None, "timeouts": tout,
               "interjections_total": interj,
               "skipped": len(progress["episodes"]) - n,
               "flags": {k: done[k] for k in sorted(done, key=int)}}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[interactive] FINAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
