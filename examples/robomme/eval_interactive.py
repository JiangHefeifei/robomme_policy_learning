"""Interactive-RoboMME minimal pipeline validation.

Tasks (easy difficulty, test split):
  ButtonUnmask  - 3 identical covers hide RGB cubes; original cue = covers lift at
                  steps 0-64 then drop back. Utterance resolves WHERE by position:
                  "first press the button, then pick up the middle container"
  PickHighlight - 3 cubes, 1 target; original cue = white disk under the target at
                  steps 10-100 then gone. Utterance resolves WHICH by color:
                  "first press the button, then pick up the red cube"
                  (episodes where a target color is duplicated on the table are
                  skipped as referentially ambiguous — same seeds => same skips
                  across conditions)

Three conditions per task:
  original  - visual cue ON, unchanged benchmark task
  no_info   - cue disabled, no utterance -> guessing floor
  utterance - cue disabled + scripted-user utterance at --args.utterance_step
              (full prompt replacement)

Reuses the running pi05 policy server exactly like eval.py, sends no subgoal keys,
never uses history. Resumes from progress.json so the known SAPIEN segfault on env
recreation is survivable by relaunching in a loop.

Run (from robomme_policy_learning, benchmark .venv python):
  PYTHONPATH=examples/robomme CUDA_VISIBLE_DEVICES=0 SAPIEN_RENDER_DEVICE=cuda \
    <benchmark python> examples/robomme/eval_interactive.py \
    --args.task=PickHighlight --args.condition=utterance --args.port=8000
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import EpisodeState, RolloutRecorder
from env_runner import EnvRunner


CONDITIONS = ("original", "no_info", "utterance")

# Which env kwarg disables the visual cue for each supported task.
TASK_DISABLE_KWARG = {
    "ButtonUnmask": "robomme_disable_lift",
    "PickHighlight": "robomme_disable_highlight",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    condition: str = "utterance"  # original | no_info | utterance
    utterance_step: int = 10
    num_episodes: int = 10        # valid (non-skipped) episodes to evaluate
    task: str = "ButtonUnmask"    # ButtonUnmask | PickHighlight
    difficulty: str = "easy"

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation_interactive"
    model_seed: int = 7           # informational (which server checkpoint seed is running)


class InteractiveEnvRunner(EnvRunner):
    """EnvRunner that forwards extra env kwargs (e.g. the cue-disable flag)."""

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
        ids = []
        for ep in range(self.env_builder.get_episode_num()):
            _, hint = self.env_builder.resolve_episode(ep)
            if hint == difficulty:
                ids.append(ep)
        return ids


def episode_is_ambiguous(task: str, env) -> bool:
    """PickHighlight only: a color-based referring expression is ambiguous if any
    target color also appears on a non-target cube. Scene layout depends only on
    the episode seed, so the same episodes are skipped in every condition."""
    if task != "PickHighlight":
        return False
    u = env.unwrapped
    return any(u.all_cube_colors.count(c) > 1 for c in u.target_cube_colors)


def scripted_user_utterance(task: str, env):
    """Build the utterance from env ground truth. Returns (utterance, meta)."""
    u = env.unwrapped

    if task == "ButtonUnmask":
        # Target is always bin_0 (it hides color_names[0], the color named in the
        # instruction). Front camera sits at +x looking down; image-right = +y,
        # so sorting containers by world y gives left-to-right order.
        ys = []
        for b in u.spawned_bins:
            p = b.pose.p
            if hasattr(p, "detach"):
                p = p.detach().cpu().numpy()
            ys.append(float(np.asarray(p).reshape(-1)[1]))
        left_to_right = sorted(range(len(ys)), key=lambda i: ys[i])
        rank = left_to_right.index(0)
        if len(ys) == 3:
            word = ("left", "middle", "right")[rank]
        else:
            word = f"{rank + 1}th from the left"
        utterance = f"first press the button, then pick up the {word} container"
        meta = {"bin_ys": ys, "target_rank_left_to_right": rank,
                "target_color": u.color_names[0]}
        return utterance, meta

    if task == "PickHighlight":
        colors = list(u.target_cube_colors)
        if len(colors) == 1:
            phrase = f"the {colors[0]} cube"
        else:
            phrase = " and ".join(f"the {c} cube" for c in colors)
        utterance = f"first press the button, then pick up {phrase}"
        meta = {"target_colors": colors, "all_colors": list(u.all_cube_colors)}
        return utterance, meta

    raise ValueError(f"no scripted user for task {task}")


def run_episode(args: Args, env_runner: InteractiveEnvRunner, video_save_dir: Path):
    client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    epstate = EpisodeState()
    pre_traj = env_runner.get_init_obs()
    task_goal = pre_traj["task_goal"]
    recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)
    print(f"task_goal: {task_goal}")

    epstate.image_buffer.extend(pre_traj["images"])
    epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
    epstate.state_buffer.extend(pre_traj["states"])
    for i in range(len(pre_traj["images"])):
        recorder.record(
            image=pre_traj["images"][i].copy(),
            wrist_image=pre_traj["wrist_images"][i].copy(),
            state=pre_traj["states"][i].copy(),
        )
    epstate.exec_start_idx = len(epstate.image_buffer) - 1

    utterance_prompt = None
    utterance_meta = None
    if args.condition == "utterance":
        utterance_prompt, utterance_meta = scripted_user_utterance(args.task, env_runner.env)
        utterance_meta = {"utterance_step": args.utterance_step,
                          "utterance": utterance_prompt, **utterance_meta}
        print(f"[scripted user] will say at step {args.utterance_step}: {utterance_prompt!r}")

    img, wrist_img, robot_state = epstate.get_current_obs()
    success_flag = "unknown"

    while True:
        if not epstate.action_plan:
            if utterance_prompt is not None and epstate.count >= args.utterance_step:
                prompt = utterance_prompt
                recorder.task_goal = "[HUMAN] " + utterance_prompt  # visible in the video
            else:
                prompt = task_goal
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
        video_filename = (f"{env_runner.env_id}_ep{env_runner.episode_id}_{args.condition}"
                          f"_{success_flag}_{env_runner.difficulty}.mp4")
        recorder.save_video(video_filename)
    return success_flag, utterance_meta


def evaluate(args: Args):
    assert args.condition in CONDITIONS, f"condition must be one of {CONDITIONS}"
    assert args.task in TASK_DISABLE_KWARG, f"task must be one of {list(TASK_DISABLE_KWARG)}"

    save_dir = Path(args.save_dir) / args.task / args.condition / f"seed{args.model_seed}"
    video_save_dir = save_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    progress_path = save_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        print(f"[interactive] resuming: {len(progress.get('episodes', {}))} episodes recorded")
    else:
        progress = {"condition": args.condition, "episodes": {}, "utterances": {}}

    extra_env_kwargs = {}
    if args.condition in ("no_info", "utterance"):
        extra_env_kwargs[TASK_DISABLE_KWARG[args.task]] = True

    env_runner = InteractiveEnvRunner(
        args.task, video_save_dir, max_steps=args.max_steps,
        extra_env_kwargs=extra_env_kwargs,
    )
    candidate_ids = env_runner.episodes_with_difficulty(args.difficulty)
    print(f"[interactive] task={args.task} condition={args.condition} "
          f"difficulty={args.difficulty} candidates={candidate_ids}")

    def n_valid():
        return sum(1 for v in progress["episodes"].values() if v != "skipped_ambiguous")

    for ep in candidate_ids:
        if n_valid() >= args.num_episodes:
            break
        if str(ep) in progress["episodes"]:
            continue
        env_runner.make_env(ep)
        if episode_is_ambiguous(args.task, env_runner.env):
            print(f"[interactive] ep{ep} skipped: target color duplicated on table")
            progress["episodes"][str(ep)] = "skipped_ambiguous"
            env_runner.close_env()
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            continue
        print(f"\n[interactive] {args.task} ep{ep} ({args.condition}) env ready, "
              f"difficulty={env_runner.difficulty}")
        flag, utt = run_episode(args, env_runner, video_save_dir)
        progress["episodes"][str(ep)] = flag
        if utt:
            progress["utterances"][str(ep)] = utt
        env_runner.close_env()
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"[interactive] ep{ep} -> {flag}")

    done = {k: v for k, v in progress["episodes"].items() if v != "skipped_ambiguous"}
    n = len(done)
    succ = sum(1 for v in done.values() if v == "success")
    summary = {"task": args.task, "condition": args.condition,
               "episodes": n, "success": succ,
               "success_rate": succ / n if n else None,
               "skipped_ambiguous": len(progress["episodes"]) - n,
               "flags": {k: done[k] for k in sorted(done, key=int)}}
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[interactive] FINAL " + json.dumps(summary))


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
