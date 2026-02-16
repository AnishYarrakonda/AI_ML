from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
import queue
import random
import threading
import time
from typing import Callable

# Keep matplotlib cache/config local to this project if user config dir is not writable.
LOCAL_MPLCONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
os.makedirs(LOCAL_MPLCONFIG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", LOCAL_MPLCONFIG)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from .game_logic import SnakeConfig, SnakeGame
except ImportError:
    from game_logic import SnakeConfig, SnakeGame


BOARD_SIZES = (10, 20, 30, 40)
APPLE_CHOICES = (1, 3, 5, 10)
ACTIONS = ("up", "down", "left", "right")
REVERSE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    board_size: int = 20
    apples: int = 3
    episodes: int = 3000
    max_steps: int = 1200
    gamma: float = 0.95
    lr: float = 0.0008
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    batch_size: int = 512
    memory_size: int = 100_000
    hidden_dim: int = 256
    target_update_every: int = 250
    step_delay: float = 0.0


class QNetwork(nn.Module):
    """Simple MLP where input dimension is board_size * board_size."""

    def __init__(self, input_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SnakeDQNAgent:
    """DQN-style agent that learns Q(state, action) with epsilon-greedy exploration."""

    def __init__(self, cfg: TrainConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.input_size = cfg.board_size * cfg.board_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(self.input_size, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(self.input_size, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=cfg.memory_size)

        self.epsilon = cfg.epsilon_start
        self.learn_steps = 0

    def valid_action_indices(self, current_direction: str) -> list[int]:
        blocked = REVERSE_DIRECTION[current_direction]
        return [idx for idx, action in enumerate(ACTIONS) if action != blocked]

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(valid_indices)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        mask = torch.full((4,), float("-inf"), device=self.device)
        mask[valid_indices] = 0.0
        return int(torch.argmax(q_values + mask).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.cfg.batch_size:
            return None

        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]

        target = rewards_t + self.cfg.gamma * max_next_q * (1.0 - dones_t)
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "board_size": self.cfg.board_size,
            "hidden_dim": self.cfg.hidden_dim,
            "state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "cfg": vars(self.cfg),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        board_size = int(payload.get("board_size", self.cfg.board_size))
        if board_size != self.cfg.board_size:
            raise ValueError(
                f"Model board size ({board_size}) does not match current board size ({self.cfg.board_size})."
            )

        self.policy_net.load_state_dict(payload["state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        self.learn_steps = int(payload.get("learn_steps", 0))


def default_model_path(board_size: int) -> str:
    return os.path.join(MODELS_DIR, f"snake_dqn_{board_size}x{board_size}.pt")


def encode_state(game: SnakeGame, board_size: int) -> np.ndarray:
    """
    Entire board is used as input (flattened):
    - empty 0.0
    - apple 0.5
    - snake body -0.5
    - snake head 1.0
    """
    board = np.zeros((board_size, board_size), dtype=np.float32)

    for x, y in game.apples:
        board[y, x] = 0.5

    for x, y in list(game.snake)[1:]:
        board[y, x] = -0.5

    head_x, head_y = game.snake[0]
    board[head_y, head_x] = 1.0

    return board.reshape(-1)


def nearest_apple_distance(game: SnakeGame) -> int:
    if not game.apples:
        return 0
    hx, hy = game.snake[0]
    return min(abs(ax - hx) + abs(ay - hy) for ax, ay in game.apples)


def make_game(cfg: TrainConfig) -> SnakeGame:
    game_cfg = SnakeConfig(
        grid_size=cfg.board_size,
        cell_size=18,
        speed_ms=80,
        apples=cfg.apples,
        initial_length=3,
        wrap_walls=False,
        show_grid=True,
    )
    return SnakeGame(game_cfg)


def run_episode(
    agent: SnakeDQNAgent,
    cfg: TrainConfig,
    train: bool = True,
    render_step: Callable[[SnakeGame, int, int, float], None] | None = None,
    stop_flag: threading.Event | None = None,
) -> tuple[int, float, int]:
    game = make_game(cfg)
    total_reward = 0.0

    for step in range(cfg.max_steps):
        if stop_flag and stop_flag.is_set():
            break

        state = encode_state(game, cfg.board_size)
        valid_actions = agent.valid_action_indices(game.direction)
        action_idx = agent.select_action(state, valid_actions, explore=train)
        action = ACTIONS[action_idx]

        old_length = len(game.snake)
        old_distance = nearest_apple_distance(game)

        game.queue_direction(action)
        alive = game.move()

        new_length = len(game.snake)
        new_distance = nearest_apple_distance(game)

        # Reward shaping for more stable learning on larger boards.
        reward = 0.01
        if not alive:
            reward = -1.0
        elif new_length > old_length:
            reward = 1.0
        elif new_distance < old_distance:
            reward += 0.03
        elif new_distance > old_distance:
            reward -= 0.03

        done = not alive
        next_state = encode_state(game, cfg.board_size)

        if train:
            agent.remember(state, action_idx, reward, next_state, done)
            agent.train_step()

        total_reward += reward

        if render_step is not None:
            render_step(game, step, new_length, agent.epsilon)

        if cfg.step_delay > 0:
            time.sleep(cfg.step_delay)

        if done:
            break

    return len(game.snake), total_reward, step + 1


def train_offline(
    cfg: TrainConfig,
    load_path: str | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
) -> SnakeDQNAgent:
    agent = SnakeDQNAgent(cfg)

    if load_path:
        agent.load(load_path)

    scores: list[float] = []
    avg_scores: list[float] = []

    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 5))
        current_line, = ax.plot([], [], label="Current Length", color="#1f77b4")
        avg_line, = ax.plot([], [], label="Average Length", color="#ff7f0e")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")
        ax.set_title("Snake Training")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)

    for ep in range(1, cfg.episodes + 1):
        score, _, _ = run_episode(agent, cfg, train=True)
        scores.append(float(score))
        avg_scores.append(float(np.mean(scores)))

        agent.decay_epsilon()

        if show_plot and (ep == 1 or ep % 10 == 0 or ep == cfg.episodes):
            x = np.arange(1, len(scores) + 1)
            current_line.set_data(x, scores)
            avg_line.set_data(x, avg_scores)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if ep % 50 == 0:
            print(
                f"Episode {ep}/{cfg.episodes} | "
                f"Length: {score:.0f} | Avg: {avg_scores[-1]:.2f} | Epsilon: {agent.epsilon:.4f}"
            )

    if save_path:
        agent.save(save_path)
    else:
        agent.save(default_model_path(cfg.board_size))

    if show_plot:
        plt.ioff()
        plt.show()

    return agent


class TrainingDashboard:
    """Tkinter dashboard to train/watch the snake agent with a live matplotlib graph."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Snake Q-Learning Trainer")
        self.root.geometry("1500x980")
        self.root.configure(bg="#101418")

        self.msg_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.cfg = TrainConfig()
        self.agent = SnakeDQNAgent(self.cfg)

        self.scores: list[float] = []
        self.avg_scores: list[float] = []
        self.last_snapshot: dict | None = None

        self._build_ui()
        self.root.after(80, self._poll_queue)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg="#101418")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        right = tk.Frame(self.root, bg="#101418")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

        controls = tk.LabelFrame(
            left,
            text="Training Controls",
            bg="#0f1720",
            fg="#e6eef7",
            font=("Helvetica", 12, "bold"),
            padx=12,
            pady=10,
        )
        controls.pack(fill="x")

        self.board_var = tk.StringVar(value="20")
        self.apple_var = tk.StringVar(value="3")
        self.episodes_var = tk.StringVar(value="3000")
        self.max_steps_var = tk.StringVar(value="1200")
        self.eps_decay_var = tk.StringVar(value="0.997")
        self.lr_var = tk.StringVar(value="0.0008")

        self._add_dropdown(controls, "Board", self.board_var, [str(v) for v in BOARD_SIZES])
        self._add_dropdown(controls, "Apples", self.apple_var, [str(v) for v in APPLE_CHOICES])
        self._add_entry(controls, "Episodes", self.episodes_var)
        self._add_entry(controls, "Max steps", self.max_steps_var)
        self._add_entry(controls, "Epsilon decay", self.eps_decay_var)
        self._add_entry(controls, "Learning rate", self.lr_var)

        btn_row = tk.Frame(controls, bg="#0f1720")
        btn_row.pack(fill="x", pady=(8, 0))

        tk.Button(btn_row, text="Train", command=self.start_training, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Watch", command=self.start_watch, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Stop", command=self.stop_worker, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Load", command=self.load_model, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Save", command=self.save_model, width=10).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            left,
            textvariable=self.status_var,
            fg="#d9e3ef",
            bg="#101418",
            font=("Helvetica", 12, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(10, 8))

        self.canvas = tk.Canvas(left, bg="#1c2229", width=700, height=700, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        fig = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Length")
        self.ax.grid(alpha=0.25)

        self.current_line, = self.ax.plot([], [], label="Current Length", color="#1f77b4")
        self.avg_line, = self.ax.plot([], [], label="Average Length", color="#ff7f0e")
        self.ax.legend(loc="upper left")

        self.plot_canvas = FigureCanvasTkAgg(fig, master=right)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_entry(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg="#0f1720")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg="#0f1720", fg="#dbe7f3", width=12, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=var, width=12, justify="center").pack(side="left")

    def _add_dropdown(self, parent: tk.Widget, label: str, var: tk.StringVar, options: list[str]) -> None:
        row = tk.Frame(parent, bg="#0f1720")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg="#0f1720", fg="#dbe7f3", width=12, anchor="w").pack(side="left")
        tk.OptionMenu(row, var, *options).pack(side="left")

    def _read_cfg_from_ui(self) -> TrainConfig:
        board_size = int(self.board_var.get())
        if board_size not in BOARD_SIZES:
            raise ValueError("Board size must be one of 10, 20, 30, 40.")

        apples = int(self.apple_var.get())
        if apples not in APPLE_CHOICES:
            raise ValueError("Apples must be one of 1, 3, 5, 10.")

        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        eps_decay = float(self.eps_decay_var.get())
        lr = float(self.lr_var.get())

        if episodes <= 0 or max_steps <= 0:
            raise ValueError("Episodes and max steps must be > 0.")
        if not (0.9 <= eps_decay <= 0.99999):
            raise ValueError("Epsilon decay should be between 0.9 and 0.99999.")
        if lr <= 0:
            raise ValueError("Learning rate must be > 0.")

        return TrainConfig(
            board_size=board_size,
            apples=apples,
            episodes=episodes,
            max_steps=max_steps,
            lr=lr,
            epsilon_decay=eps_decay,
        )

    def _reset_agent_for_cfg(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.agent = SnakeDQNAgent(cfg)

    def _draw_snapshot(self, snapshot: dict) -> None:
        size = snapshot["size"]
        snake = snapshot["snake"]
        apples = snapshot["apples"]

        canvas_w = max(self.canvas.winfo_width(), 200)
        canvas_h = max(self.canvas.winfo_height(), 200)
        cell = max(8, min(canvas_w, canvas_h) // size)

        board_w = cell * size
        board_h = cell * size
        x_off = (canvas_w - board_w) // 2
        y_off = (canvas_h - board_h) // 2

        self.canvas.delete("all")

        for i in range(size + 1):
            pos = i * cell
            self.canvas.create_line(x_off, y_off + pos, x_off + board_w, y_off + pos, fill="#2a3340")
            self.canvas.create_line(x_off + pos, y_off, x_off + pos, y_off + board_h, fill="#2a3340")

        self.canvas.create_rectangle(x_off, y_off, x_off + board_w, y_off + board_h, outline="#7f8b99", width=2)

        for x, y in apples:
            self.canvas.create_oval(
                x_off + x * cell + 3,
                y_off + y * cell + 3,
                x_off + (x + 1) * cell - 3,
                y_off + (y + 1) * cell - 3,
                fill="#ff5c74",
                outline="",
            )

        for idx, (x, y) in enumerate(snake):
            color = "#45d483" if idx == 0 else "#1fb86b"
            self.canvas.create_rectangle(
                x_off + x * cell + 2,
                y_off + y * cell + 2,
                x_off + (x + 1) * cell - 2,
                y_off + (y + 1) * cell - 2,
                fill=color,
                outline="",
            )

    def _update_plot(self) -> None:
        if not self.scores:
            return
        x = np.arange(1, len(self.scores) + 1)
        self.current_line.set_data(x, self.scores)
        self.avg_line.set_data(x, self.avg_scores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.plot_canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                mtype = msg["type"]

                if mtype == "step":
                    self.last_snapshot = msg["snapshot"]
                    self._draw_snapshot(self.last_snapshot)

                elif mtype == "episode":
                    score = float(msg["score"])
                    avg = float(msg["avg"])
                    epsilon = float(msg["epsilon"])
                    episode = int(msg["episode"])
                    total = int(msg["total"])

                    self.scores.append(score)
                    self.avg_scores.append(avg)
                    self._update_plot()
                    self.status_var.set(
                        f"Episode {episode}/{total} | Length: {score:.0f} | Avg: {avg:.2f} | Epsilon: {epsilon:.4f}"
                    )

                elif mtype == "done":
                    self.status_var.set(msg.get("text", "Finished"))
                    self.worker = None

                elif mtype == "error":
                    self.worker = None
                    messagebox.showerror("Training Error", msg["text"])
                    self.status_var.set("Error")
        except queue.Empty:
            pass

        self.root.after(80, self._poll_queue)

    def _launch_worker(self, fn: Callable[[], None]) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Training/watch is already running.")
            return

        self.stop_event.clear()
        self.worker = threading.Thread(target=fn, daemon=True)
        self.worker.start()

    def start_training(self) -> None:
        try:
            cfg = self._read_cfg_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        if cfg.board_size != self.cfg.board_size:
            self._reset_agent_for_cfg(cfg)
        else:
            self.cfg = cfg
            self.agent.cfg = cfg

        self.scores.clear()
        self.avg_scores.clear()
        self._update_plot()

        def worker() -> None:
            try:
                running_sum = 0.0

                def on_step(game: SnakeGame, _step: int, _length: int, _eps: float) -> None:
                    if self.stop_event.is_set():
                        return
                    self.msg_queue.put(
                        {
                            "type": "step",
                            "snapshot": {
                                "size": cfg.board_size,
                                "snake": list(game.snake),
                                "apples": list(game.apples),
                            },
                        }
                    )

                for ep in range(1, cfg.episodes + 1):
                    if self.stop_event.is_set():
                        break

                    score, _, _ = run_episode(
                        self.agent,
                        cfg,
                        train=True,
                        render_step=on_step,
                        stop_flag=self.stop_event,
                    )
                    self.agent.decay_epsilon()

                    running_sum += score
                    avg = running_sum / ep

                    self.msg_queue.put(
                        {
                            "type": "episode",
                            "episode": ep,
                            "total": cfg.episodes,
                            "score": score,
                            "avg": avg,
                            "epsilon": self.agent.epsilon,
                        }
                    )

                self.msg_queue.put({"type": "done", "text": "Training complete"})
            except Exception as exc:
                self.msg_queue.put({"type": "error", "text": str(exc)})

        self._launch_worker(worker)

    def start_watch(self) -> None:
        try:
            cfg = self._read_cfg_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        if cfg.board_size != self.cfg.board_size:
            self._reset_agent_for_cfg(cfg)

        self.status_var.set("Watching model play...")

        def worker() -> None:
            try:
                watch_cfg = cfg
                watch_cfg.step_delay = 0.05

                saved_eps = self.agent.epsilon
                self.agent.epsilon = 0.0

                for ep in range(1, 500000):
                    if self.stop_event.is_set():
                        break

                    def on_step(game: SnakeGame, _step: int, _length: int, _eps: float) -> None:
                        if self.stop_event.is_set():
                            return
                        self.msg_queue.put(
                            {
                                "type": "step",
                                "snapshot": {
                                    "size": watch_cfg.board_size,
                                    "snake": list(game.snake),
                                    "apples": list(game.apples),
                                },
                            }
                        )

                    score, _, _ = run_episode(
                        self.agent,
                        watch_cfg,
                        train=False,
                        render_step=on_step,
                        stop_flag=self.stop_event,
                    )

                    self.msg_queue.put(
                        {
                            "type": "episode",
                            "episode": ep,
                            "total": ep,
                            "score": score,
                            "avg": score if not self.avg_scores else (sum(self.scores + [score]) / (len(self.scores) + 1)),
                            "epsilon": self.agent.epsilon,
                        }
                    )

                self.agent.epsilon = saved_eps
                self.msg_queue.put({"type": "done", "text": "Watch stopped"})
            except Exception as exc:
                self.msg_queue.put({"type": "error", "text": str(exc)})

        self._launch_worker(worker)

    def stop_worker(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def save_model(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pt",
            initialfile=f"snake_dqn_{self.cfg.board_size}x{self.cfg.board_size}.pt",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.agent.save(path)
            self.status_var.set(f"Saved model: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            payload = torch.load(path, map_location="cpu")
            board_size = int(payload.get("board_size", self.cfg.board_size))
            if board_size not in BOARD_SIZES:
                raise ValueError(f"Unsupported board size in model: {board_size}")

            # Automatically switch dashboard board size to the loaded model.
            self.board_var.set(str(board_size))
            cfg = self._read_cfg_from_ui()
            self._reset_agent_for_cfg(cfg)
            self.agent.load(path)
            self.status_var.set(f"Loaded model: {path}")
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))


def launch_training_gui() -> None:
    root = tk.Tk()
    TrainingDashboard(root)
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake Q-learning/DQN trainer")
    parser.add_argument("--mode", choices=["train", "train-gui"], default="train-gui")
    parser.add_argument("--board-size", type=int, default=20, choices=BOARD_SIZES)
    parser.add_argument("--apples", type=int, default=3, choices=APPLE_CHOICES)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib live plot in offline mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train-gui":
        launch_training_gui()
        return

    cfg = TrainConfig(
        board_size=args.board_size,
        apples=args.apples,
        episodes=args.episodes,
        max_steps=args.max_steps,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        lr=args.lr,
    )

    save_path = args.save or default_model_path(cfg.board_size)
    load_path = args.load if args.load else None

    train_offline(
        cfg,
        load_path=load_path,
        save_path=save_path,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
