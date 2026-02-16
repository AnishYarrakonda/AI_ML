from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

# Support both package imports and running this file directly.
try:
    from .game_logic import (
        MAX_APPLES,
        MAX_CELL_SIZE,
        MAX_GRID_SIZE,
        MAX_SPEED_MS,
        MIN_APPLES,
        MIN_CELL_SIZE,
        MIN_GRID_SIZE,
        MIN_INITIAL_LENGTH,
        MIN_SPEED_MS,
        SnakeConfig,
        SnakeGame,
    )
except ImportError:
    from game_logic import (
        MAX_APPLES,
        MAX_CELL_SIZE,
        MAX_GRID_SIZE,
        MAX_SPEED_MS,
        MIN_APPLES,
        MIN_CELL_SIZE,
        MIN_GRID_SIZE,
        MIN_INITIAL_LENGTH,
        MIN_SPEED_MS,
        SnakeConfig,
        SnakeGame,
    )


class SnakeApp:
    """Tkinter presentation layer for SnakeGame."""
    UI_SCALE = 1.35
    BG = "#101418"
    BOARD_BG = "#1c2229"
    SIDEBAR_BG = "#0f1720"
    GRID_COLOR = "#293340"
    SNAKE_HEAD = "#45d483"
    SNAKE_BODY = "#1fb86b"
    APPLE_COLOR = "#ff5c74"
    TEXT_PRIMARY = "#e6eef7"
    TEXT_MUTED = "#95a4b8"
    ACCENT = "#42c4ff"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Snake Trainer")
        self.root.configure(bg=self.BG)
        self.root.tk.call("tk", "scaling", self.UI_SCALE)
        self.root.minsize(self._s(1200), self._s(860))
        self.root.geometry(f"{self._s(1480)}x{self._s(980)}")

        self.config = SnakeConfig()
        self.game = SnakeGame(self.config)
        self.after_id: str | None = None  # Tkinter timer id for the game loop

        self._build_layout()
        self._bind_keys()
        self._apply_canvas_size()
        self.draw()

    def _s(self, value: int) -> int:
        """Scale pixel/font values for better readability."""
        return int(round(value * self.UI_SCALE))

    def _build_layout(self) -> None:
        """Create game canvas + right sidebar panels."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = tk.Frame(self.root, bg=self.BG)
        container.grid(row=0, column=0, sticky="nsew", padx=self._s(16), pady=self._s(16))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container,
            bg=self.BOARD_BG,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, self._s(16)))

        self.sidebar = tk.Frame(container, bg=self.SIDEBAR_BG, width=self._s(430))
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(False)

        title = tk.Label(
            self.sidebar,
            text="Snake Controls",
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            font=("Helvetica", self._s(16), "bold"),
        )
        title.pack(anchor="w", padx=self._s(16), pady=(self._s(16), self._s(6)))

        subtitle = tk.Label(
            self.sidebar,
            text="Tune settings, then press Start",
            fg=self.TEXT_MUTED,
            bg=self.SIDEBAR_BG,
            font=("Helvetica", self._s(10)),
        )
        subtitle.pack(anchor="w", padx=self._s(16), pady=(0, self._s(14)))

        self._build_status()
        self._build_controls()
        self._build_buttons()

    def _build_status(self) -> None:
        """Top sidebar section with live score/length/run-state labels."""
        frame = tk.LabelFrame(
            self.sidebar,
            text="Status",
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            bd=1,
            font=("Helvetica", self._s(10), "bold"),
            labelanchor="n",
        )
        frame.pack(fill="x", padx=self._s(16), pady=(0, self._s(14)))

        self.score_var = tk.StringVar(value="Score: 0")
        self.length_var = tk.StringVar(value=f"Length: {len(self.game.snake)}")
        self.state_var = tk.StringVar(value="State: Ready")

        for var in (self.score_var, self.length_var, self.state_var):
            tk.Label(
                frame,
                textvariable=var,
                fg=self.TEXT_PRIMARY,
                bg=self.SIDEBAR_BG,
                font=("Helvetica", self._s(11)),
                anchor="w",
            ).pack(fill="x", padx=self._s(10), pady=self._s(4))

    def _build_controls(self) -> None:
        """Settings section for values that rebuild the game state."""
        frame = tk.LabelFrame(
            self.sidebar,
            text="Settings",
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            bd=1,
            font=("Helvetica", self._s(10), "bold"),
            labelanchor="n",
        )
        frame.pack(fill="x", padx=self._s(16), pady=(0, self._s(14)))

        self.grid_size_var = tk.StringVar(value=str(self.config.grid_size))
        self.cell_size_var = tk.StringVar(value=str(self.config.cell_size))
        self.speed_var = tk.StringVar(value=str(self.config.speed_ms))
        self.apples_var = tk.StringVar(value=str(self.config.apples))
        self.length_setting_var = tk.StringVar(value=str(self.config.initial_length))
        self.wrap_var = tk.BooleanVar(value=self.config.wrap_walls)
        self.show_grid_var = tk.BooleanVar(value=self.config.show_grid)

        self._add_labeled_spinbox(frame, "Grid Size", self.grid_size_var)
        self._add_labeled_spinbox(frame, "Cell Size", self.cell_size_var)
        self._add_labeled_spinbox(frame, "Speed (ms)", self.speed_var)
        self._add_labeled_spinbox(frame, "Apples", self.apples_var)
        self._add_labeled_spinbox(frame, "Initial Length", self.length_setting_var)

        tk.Checkbutton(
            frame,
            text="Wrap through walls",
            variable=self.wrap_var,
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            selectcolor=self.SIDEBAR_BG,
            activebackground=self.SIDEBAR_BG,
            activeforeground=self.TEXT_PRIMARY,
            font=("Helvetica", self._s(10)),
        ).pack(anchor="w", padx=self._s(10), pady=(self._s(8), self._s(2)))

        tk.Checkbutton(
            frame,
            text="Show grid lines",
            variable=self.show_grid_var,
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            selectcolor=self.SIDEBAR_BG,
            activebackground=self.SIDEBAR_BG,
            activeforeground=self.TEXT_PRIMARY,
            font=("Helvetica", self._s(10)),
        ).pack(anchor="w", padx=self._s(10), pady=(self._s(2), self._s(10)))

        hint = tk.Label(
            frame,
            text=(
                f"Ranges: grid {MIN_GRID_SIZE}-{MAX_GRID_SIZE}, cell {MIN_CELL_SIZE}-{MAX_CELL_SIZE}, "
                f"speed {MIN_SPEED_MS}-{MAX_SPEED_MS}, apples {MIN_APPLES}-{MAX_APPLES}"
            ),
            fg=self.TEXT_MUTED,
            bg=self.SIDEBAR_BG,
            justify="left",
            wraplength=self._s(320),
            font=("Helvetica", self._s(9)),
        )
        hint.pack(anchor="w", padx=self._s(10), pady=(0, self._s(10)))

    def _add_labeled_spinbox(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        """Small helper to render one setting row."""
        row = tk.Frame(parent, bg=self.SIDEBAR_BG)
        row.pack(fill="x", padx=self._s(10), pady=self._s(4))

        tk.Label(
            row,
            text=label,
            fg=self.TEXT_PRIMARY,
            bg=self.SIDEBAR_BG,
            font=("Helvetica", self._s(10)),
        ).pack(side="left")

        spin = tk.Spinbox(
            row,
            from_=0,
            to=999,
            textvariable=var,
            width=10,
            justify="center",
            bd=0,
            relief="flat",
            bg="#e8eef5",
            fg="#1a2734",
            font=("Helvetica", self._s(10)),
        )
        spin.pack(side="right")

    def _build_buttons(self) -> None:
        """Action buttons for start/pause/reset/apply."""
        frame = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG)
        frame.pack(fill="x", padx=self._s(16), pady=(0, self._s(10)))

        self.start_btn = self._button(frame, "Start", self.start_game)
        self.start_btn.pack(fill="x", pady=self._s(4))

        self.pause_btn = self._button(frame, "Pause", self.toggle_pause)
        self.pause_btn.pack(fill="x", pady=self._s(4))

        self.reset_btn = self._button(frame, "Reset", self.reset_game)
        self.reset_btn.pack(fill="x", pady=self._s(4))

        self.apply_btn = self._button(frame, "Apply Settings", self.apply_settings)
        self.apply_btn.pack(fill="x", pady=self._s(4))

        footer = tk.Label(
            self.sidebar,
            text="Move: Arrow keys / WASD",
            fg=self.TEXT_MUTED,
            bg=self.SIDEBAR_BG,
            font=("Helvetica", self._s(10)),
        )
        footer.pack(anchor="w", padx=self._s(16), pady=(self._s(4), self._s(10)))

    def _button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            fg="#09141f",
            bg=self.ACCENT,
            activebackground="#74d8ff",
            activeforeground="#09141f",
            bd=0,
            relief="flat",
            font=("Helvetica", self._s(11), "bold"),
            padx=self._s(12),
            pady=self._s(9),
            cursor="hand2",
        )

    def _bind_keys(self) -> None:
        """Bind movement controls and spacebar pause."""
        self.root.bind("<Up>", lambda _e: self.game.queue_direction("up"))
        self.root.bind("<Down>", lambda _e: self.game.queue_direction("down"))
        self.root.bind("<Left>", lambda _e: self.game.queue_direction("left"))
        self.root.bind("<Right>", lambda _e: self.game.queue_direction("right"))
        self.root.bind("w", lambda _e: self.game.queue_direction("up"))
        self.root.bind("s", lambda _e: self.game.queue_direction("down"))
        self.root.bind("a", lambda _e: self.game.queue_direction("left"))
        self.root.bind("d", lambda _e: self.game.queue_direction("right"))
        self.root.bind("<space>", lambda _e: self.toggle_pause())

    def _parse_int(self, raw: str, low: int, high: int, label: str) -> int:
        """Parse and clamp-check integer settings with a clear error message."""
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{label} must be an integer.")
        if not (low <= value <= high):
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    def apply_settings(self) -> None:
        """Validate sidebar values, then rebuild the game with new config."""
        try:
            grid_size = self._parse_int(self.grid_size_var.get(), MIN_GRID_SIZE, MAX_GRID_SIZE, "Grid size")
            cell_size = self._parse_int(self.cell_size_var.get(), MIN_CELL_SIZE, MAX_CELL_SIZE, "Cell size")
            speed_ms = self._parse_int(self.speed_var.get(), MIN_SPEED_MS, MAX_SPEED_MS, "Speed")
            apples = self._parse_int(self.apples_var.get(), MIN_APPLES, MAX_APPLES, "Apples")
            initial_length = self._parse_int(
                self.length_setting_var.get(),
                MIN_INITIAL_LENGTH,
                grid_size,
                "Initial length",
            )
        except ValueError as exc:
            messagebox.showerror("Invalid Setting", str(exc))
            return

        # Keep at least one free tile for movement/spawns.
        max_apples_by_tiles = max(1, grid_size * grid_size - initial_length)
        if apples > max_apples_by_tiles:
            messagebox.showerror(
                "Invalid Setting",
                (
                    f"Apples is too high for the selected grid and snake length. "
                    f"Maximum allowed is {max_apples_by_tiles}."
                ),
            )
            return

        self.config = SnakeConfig(
            grid_size=grid_size,
            cell_size=cell_size,
            speed_ms=speed_ms,
            apples=apples,
            initial_length=initial_length,
            wrap_walls=self.wrap_var.get(),
            show_grid=self.show_grid_var.get(),
        )
        self.game = SnakeGame(self.config)
        self._cancel_loop()
        self._apply_canvas_size()
        self.state_var.set("State: Ready")
        self.draw()

    def _apply_canvas_size(self) -> None:
        """Resize board canvas to match current grid + tile size."""
        side_pixels = self.config.grid_size * self.config.cell_size
        self.canvas.configure(width=side_pixels, height=side_pixels)

    def _cancel_loop(self) -> None:
        """Cancel scheduled tick callback if one exists."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def start_game(self) -> None:
        """Start or resume live ticking from current state."""
        if not self.game.alive:
            self.game.reset()
        self.game.running = True
        self.state_var.set("State: Running")
        self.tick()

    def toggle_pause(self) -> None:
        """Pause/resume without losing current board state."""
        if not self.game.alive:
            return
        self.game.running = not self.game.running
        if self.game.running:
            self.state_var.set("State: Running")
            self.tick()
        else:
            self.state_var.set("State: Paused")
            self._cancel_loop()

    def reset_game(self) -> None:
        """Reset state using current config values."""
        self._cancel_loop()
        self.game = SnakeGame(self.config)
        self.state_var.set("State: Ready")
        self.draw()

    def tick(self) -> None:
        """Single frame of the game loop; reschedules itself while running."""
        self._cancel_loop()
        if not self.game.running:
            return

        if not self.game.move():
            self.game.running = False
            self.state_var.set("State: Game Over")
            self.draw()
            return

        self.draw()
        self.after_id = self.root.after(self.config.speed_ms, self.tick)

    def draw(self) -> None:
        """Render board, apples, snake, status labels, and game-over overlay."""
        self.canvas.delete("all")
        size = self.config.grid_size
        cell = self.config.cell_size

        if self.config.show_grid:
            # Optional guide lines to make large boards easier to read.
            for i in range(size + 1):
                pos = i * cell
                self.canvas.create_line(0, pos, size * cell, pos, fill=self.GRID_COLOR)
                self.canvas.create_line(pos, 0, pos, size * cell, fill=self.GRID_COLOR)

        for x, y in self.game.apples:
            x1, y1 = x * cell + 4, y * cell + 4
            x2, y2 = (x + 1) * cell - 4, (y + 1) * cell - 4
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.APPLE_COLOR, outline="")

        for idx, (x, y) in enumerate(self.game.snake):
            color = self.SNAKE_HEAD if idx == 0 else self.SNAKE_BODY
            x1, y1 = x * cell + 2, y * cell + 2
            x2, y2 = (x + 1) * cell - 2, (y + 1) * cell - 2
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        self.score_var.set(f"Score: {self.game.score}")
        self.length_var.set(f"Length: {len(self.game.snake)}")

        if not self.game.alive:
            side = size * cell
            self.canvas.create_rectangle(0, 0, side, side, fill="#000000", stipple="gray50", outline="")
            self.canvas.create_text(
                side // 2,
                side // 2 - 12,
                text="Game Over",
                fill=self.TEXT_PRIMARY,
                font=("Helvetica", 22, "bold"),
            )
            self.canvas.create_text(
                side // 2,
                side // 2 + 20,
                text="Press Reset or Start",
                fill=self.TEXT_MUTED,
                font=("Helvetica", 12),
            )


def main() -> None:
    """Launch the desktop app."""
    root = tk.Tk()
    SnakeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
