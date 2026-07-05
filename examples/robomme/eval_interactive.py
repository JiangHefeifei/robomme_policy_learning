"""Interactive-RoboMME zero-shot probe: all seven interaction types on a frozen
pi05_baseline policy (no training, original scorers untouched).

Tasks x conditions (easy difficulty, test split):

PickHighlight (color language — high zero-shot compliance):
  original    visual highlight ON (benchmark as-is)
  no_info     highlight OFF, no utterance (floor)
  utterance   highlight OFF, told at step 10                      [type 1: told]
  reactive    highlight OFF, vague prompt; when TCP approaches a
              NON-target cube -> "No, not that one — pick up the
              {color} cube."                                      [type 2: reactive correction]
  update      highlight OFF, told WRONG color at step 10, corrected
              at step 150 ("Actually, ... instead")               [type 6: mid-course change]
  prohibition highlight OFF, only told what NOT to touch          [type 7: prohibition]
  ask         highlight OFF, vague prompt; if still running at step
              K the robot ASKS, scripted user answers by color,
              answer injected + stored to memory_store.json       [type 5/ask: uncertainty-aware question]
  ask_memory  highlight OFF; same layouts as `ask`, answer replayed
              from memory at step 10, ZERO questions              [ask-or-remember closed loop]

ButtonUnmask (spatial language — known-weak grounding):
  original / no_info / utterance  (already measured)
  ask / ask_memory                same stuck-triggered gate as above

MoveCube (manner selection):                                      [type 3: better way]
  prior       no utterance; policy acts on its own prior (the
              no-history baseline never sees the demo video)
  told        the sampled manner is told at step 10

BinFill (standing preference as default params):                  [type 4: preference]
  original    true instruction from step 0 (anchor)
  vague       vague instruction all episode (floor)
  told        vague until step 10, then the true preference is told

Uncertainty signal (v0, zero-shot): a stuck detector — episode still running at
step K with no info given => the gate fires one question. Honest limitation: this
is a time-based proxy, not model introspection (flow matching has no token
probabilities); ensembling/retrieval-confidence gates are method-stage work.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import EpisodeState, RolloutRecorder
from env_runner import EnvRunner


# (task, condition) -> env kwarg that disables the visual cue (None = no change)
CUE_OFF = {
    "ButtonUnmask": "robomme_disable_lift",
    "PickHighlight": "robomme_disable_highlight",
    "MoveCube": None,
    "BinFill": None,
}

CONDITIONS = {
    "PickHighlight": ["original", "no_info", "utterance", "reactive", "update",
                      "prohibition", "ask", "ask_memory"],
    "ButtonUnmask": ["original", "no_info", "utterance", "ask", "ask_memory"],
    "MoveCube": ["prior", "told"],
    "BinFill": ["original", "vague", "told"],
}

# conditions that run with the visual cue ON / unchanged env
CUE_ON_CONDITIONS = {"original", "prior", "vague", "told"}

VAGUE_PROMPTS = {
    "PickHighlight": "first press the button, then pick up the cube I want",
    "ButtonUnmask": "first press the button, then pick up the container I want",
    "BinFill": "put the cubes I like into the bin, then press the button to stop",
}

MOVECUBE_WAY_PROMPT = {
    "peg_push": "pick up the peg and use it to push the cube onto the target - do not grasp the cube",
    "gripper_push": "push the cube onto the target with your gripper - do not grasp the cube",
    "grasp_putdown": "pick up the cube and place it down on the target",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    task: str = "PickHighlight"
    condition: str = "utterance"
    utterance_step: int = 10      # told / update(first) / prohibition / ask_memory replay
    update_step: int = 150        # update: when the correction arrives
    ask_step: int = 100           # ask: stuck/no-info detector threshold (fires before
                                  # the policy typically commits to a wrong grasp)
    proximity_m: float = 0.10     # reactive: TCP-to-non-target trigger distance

    num_episodes: int = 10
    difficulty: str = "easy"
    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation_interactive"
    model_seed: int = 7


class InteractiveEnvRunner(EnvRunner):
    def __init__(self, env_id, video_save_dir, max_steps=1300, extra_env_kwargs=None):
        super().__init__(env_id, video_save_dir, max_steps=max_steps)
        self.extra_env_kwargs = dict(extra_env_kwargs or {})

    def make_env(self, episode_id: int) -> None:
        self.env = self.env_builder.make_env_for_episode(
            episode_id, extra_env_kwargs=self.extra_env_kwargs
        )
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty

    def episodes_with_difficulty(self, difficulty: str):
        return [ep for ep in range(self.env_builder.get_episode_num())
                if self.env_builder.resolve_episode(ep)[1] == difficulty]


# ---------------------------------------------------------------- ground truth

def _pos(obj):
    """World position of an actor (has .pose) or a raw Pose (has .p)."""
    p = obj.pose.p if hasattr(obj, "pose") else obj.p
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p).reshape(-1)[:3]


def target_color(u):
    return u.target_cube_colors[0]


def nontarget_colors(u):
    return [c for c in u.all_cube_colors if c not in u.target_cube_colors]


def target_color_unique(u) -> bool:
    return all(u.all_cube_colors.count(c) == 1 for c in u.target_cube_colors)


def all_colors_unique(u) -> bool:
    return len(set(u.all_cube_colors)) == len(u.all_cube_colors)


def bin_position_word(u) -> str:
    """ButtonUnmask: target is always bin_0; image-right = +y."""
    ys = [float(_pos(b)[1]) for b in u.spawned_bins]
    rank = sorted(range(len(ys)), key=lambda i: ys[i]).index(0)
    return ("left", "middle", "right")[rank] if len(ys) == 3 else f"{rank + 1}th from the left"


def resolved_instruction(task: str, u) -> str:
    """The fully-resolved instruction a cooperative human would give."""
    if task == "PickHighlight":
        return f"first press the button, then pick up the {target_color(u)} cube"
    if task == "ButtonUnmask":
        return f"first press the button, then pick up the {bin_position_word(u)} container"
    raise ValueError(task)


# ------------------------------------------------------- per-episode controller

class Controller:
    """Decides the prompt at every chunk boundary; fires trigger-based events."""

    def __init__(self, args: Args, task_goal: str, env, memory_store: dict):
        self.args = args
        self.c = args.condition
        self.task = args.task
        self.env = env
        self.u = env.unwrapped
        self.task_goal = task_goal
        self.memory_store = memory_store
        self.events = []          # [{step, kind, text}]
        self.n_questions = 0
        self._pending = None      # prompt override once set
        self._fired = set()

        t = self.task
        if self.c == "utterance":
            self._plan = [(args.utterance_step, "tell", resolved_instruction(t, self.u))]
        elif self.c == "update":
            decoy = nontarget_colors(self.u)[0]
            self._plan = [
                (args.utterance_step, "tell",
                 f"first press the button, then pick up the {decoy} cube"),
                (args.update_step, "update",
                 f"actually, do not pick the {decoy} cube - "
                 f"pick up the {target_color(self.u)} cube instead"),
            ]
        elif self.c == "prohibition":
            c1, c2 = nontarget_colors(self.u)[:2]
            self._plan = [(args.utterance_step, "tell",
                           f"first press the button, then pick up a cube, "
                           f"but do not touch the {c1} cube and do not touch the {c2} cube")]
        elif self.c == "told" and t == "MoveCube":
            self._plan = [(args.utterance_step, "tell", MOVECUBE_WAY_PROMPT[self.u.way])]
        elif self.c == "told" and t == "BinFill":
            self._plan = [(args.utterance_step, "tell", task_goal)]  # true preference told at t
        elif self.c == "ask_memory":
            remembered = self.memory_store.get(t, {}).get(str(getattr(env, "episode_id", "")), None)
            self._remembered = remembered
            self._plan = ([(args.utterance_step, "memory", remembered)]
                          if remembered else [])
        else:
            self._plan = []

    def initial_prompt(self) -> str:
        if self.c in ("reactive", "ask", "ask_memory") and self.task in VAGUE_PROMPTS:
            return VAGUE_PROMPTS[self.task]
        if self.c == "vague":
            return VAGUE_PROMPTS[self.task]
        if self.c == "told" and self.task == "BinFill":
            return VAGUE_PROMPTS["BinFill"]
        return self.task_goal

    def _fire(self, step, kind, text):
        key = (kind, text)
        if key in self._fired:
            return
        self._fired.add(key)
        self._pending = text
        self.events.append({"step": int(step), "kind": kind, "text": text})
        print(f"[scripted user] step {step} [{kind}]: {text!r}")

    def on_step(self, step: int):
        # scheduled utterances
        for (t, kind, text) in self._plan:
            if text and step >= t:
                self._fire(t, kind, text)

        # reactive: TCP approaches a non-target cube
        if self.c == "reactive" and self.task == "PickHighlight":
            tcp = _pos(self.u.agent.tcp_pose)
            nts = [cube for cube in self.u.all_cubes if cube not in self.u.target_cubes]
            if nts:
                d = min(float(np.linalg.norm(tcp - _pos(cb))) for cb in nts)
                if d < self.args.proximity_m:
                    self._fire(step, "correction",
                               f"no, not that one - pick up the {target_color(self.u)} cube")

        # ask: stuck-detector question
        if self.c == "ask" and step >= self.args.ask_step and not getattr(self, "_asked", False):
            self._asked = True
            answer = resolved_instruction(self.task, self.u)
            question = ("which cube do you want?" if self.task == "PickHighlight"
                        else "which container should I pick up?")
            self.n_questions += 1
            self.events.append({"step": int(step), "kind": "question", "text": question})
            print(f"[robot asks] step {step}: {question!r}")
            self._fire(step, "answer", answer)

    def current_prompt(self) -> str:
        return self._pending if self._pending is not None else self.initial_prompt()

    def episode_meta(self):
        meta = {"events": self.events, "n_questions": self.n_questions}
        if self.c == "ask":
            # persist the answer for the ask_memory pass
            for e in self.events:
                if e["kind"] == "answer":
                    meta["stored_answer"] = e["text"]
        if self.c == "ask_memory":
            meta["memory_hit"] = bool(getattr(self, "_remembered", None))
        if self.task == "MoveCube":
            meta["sampled_way"] = self.u.way
        return meta


def episode_skip_reason(args: Args, u) -> str:
    """Referential-ambiguity screens (seed-determined => identical across conditions)."""
    if args.task != "PickHighlight":
        return ""
    if args.condition in ("update", "prohibition"):
        if not all_colors_unique(u):
            return "skipped_ambiguous"
    elif not target_color_unique(u):
        return "skipped_ambiguous"
    return ""


# ------------------------------------------------------------------- main loop

def run_episode(args: Args, env_runner, video_save_dir: Path, memory_store: dict):
    client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    epstate = EpisodeState()
    pre_traj = env_runner.get_init_obs()
    task_goal = pre_traj["task_goal"]

    env_runner.env.episode_id = env_runner.episode_id  # for ask_memory lookup
    ctrl = Controller(args, task_goal, env_runner.env, memory_store)

    recorder = RolloutRecorder(video_save_dir, ctrl.initial_prompt(), fps=30)
    print(f"task_goal: {task_goal}  |  initial prompt: {ctrl.initial_prompt()!r}")

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
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": robot_state,
                "prompt": prompt,
            }
            action_chunk = client.infer(element)["actions"]
            epstate.action_plan.extend(action_chunk[: args.obs_horizon])
            epstate.clear_buffers()

        action = epstate.action_plan.popleft()
        obs, stop_flag, success_flag = env_runner.step(action)
        epstate.count += 1
        ctrl.on_step(epstate.count)

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
        _pad_frames_to_same_size(recorder)  # prompt text area height varies with
        # utterance length -> frame sizes can differ within one episode, which
        # imageio rejects ("All images in a movie should have same size")
        recorder.save_video(f"{env_runner.env_id}_ep{env_runner.episode_id}_"
                            f"{args.condition}_{success_flag}_{env_runner.difficulty}.mp4")
    return success_flag, ctrl.episode_meta()


def _pad_frames_to_same_size(recorder):
    frames = recorder.total_images
    if not frames:
        return
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    for i, f in enumerate(frames):
        if f.shape[0] != h or f.shape[1] != w:
            canvas = np.zeros((h, w, f.shape[2]), dtype=f.dtype)
            canvas[: f.shape[0], : f.shape[1]] = f
            frames[i] = canvas


def evaluate(args: Args):
    assert args.task in CONDITIONS, f"task must be one of {list(CONDITIONS)}"
    assert args.condition in CONDITIONS[args.task], \
        f"{args.task} supports {CONDITIONS[args.task]}"

    save_dir = Path(args.save_dir) / args.task / args.condition / f"seed{args.model_seed}"
    video_save_dir = save_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    memory_path = Path(args.save_dir) / "memory_store.json"
    memory_store = json.loads(memory_path.read_text()) if memory_path.exists() else {}

    progress_path = save_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        print(f"[interactive] resuming: {len(progress.get('episodes', {}))} recorded")
    else:
        progress = {"condition": args.condition, "episodes": {}, "meta": {}}

    extra = {}
    cue_kwarg = CUE_OFF.get(args.task)
    if cue_kwarg and args.condition not in CUE_ON_CONDITIONS:
        extra[cue_kwarg] = True

    env_runner = InteractiveEnvRunner(args.task, video_save_dir,
                                      max_steps=args.max_steps, extra_env_kwargs=extra)
    candidates = env_runner.episodes_with_difficulty(args.difficulty)
    print(f"[interactive] {args.task}/{args.condition} candidates={candidates}")

    def n_valid():
        return sum(1 for v in progress["episodes"].values() if v != "skipped_ambiguous")

    for ep in candidates:
        if n_valid() >= args.num_episodes:
            break
        if str(ep) in progress["episodes"]:
            continue
        env_runner.make_env(ep)
        reason = episode_skip_reason(args, env_runner.env.unwrapped)
        if reason:
            print(f"[interactive] ep{ep} {reason}")
            progress["episodes"][str(ep)] = reason
            env_runner.close_env()
            progress_path.write_text(json.dumps(progress, indent=2))
            continue
        print(f"\n[interactive] {args.task} ep{ep} ({args.condition})")
        flag, meta = run_episode(args, env_runner, video_save_dir, memory_store)
        progress["episodes"][str(ep)] = flag
        progress["meta"][str(ep)] = meta
        env_runner.close_env()

        if args.condition == "ask" and meta.get("stored_answer"):
            memory_store.setdefault(args.task, {})[str(ep)] = meta["stored_answer"]
            memory_path.write_text(json.dumps(memory_store, indent=2))

        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"[interactive] ep{ep} -> {flag}")

    done = {k: v for k, v in progress["episodes"].items() if v != "skipped_ambiguous"}
    n = len(done)
    succ = sum(1 for v in done.values() if v == "success")
    tout = sum(1 for v in done.values() if v == "timeout")
    qs = sum(progress["meta"].get(k, {}).get("n_questions", 0) for k in done)
    summary = {"task": args.task, "condition": args.condition,
               "episodes": n, "success": succ,
               "success_rate": succ / n if n else None,
               "timeouts": tout, "questions_total": qs,
               "skipped_ambiguous": len(progress["episodes"]) - n,
               "flags": {k: done[k] for k in sorted(done, key=int)}}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[interactive] FINAL " + json.dumps(summary))


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
