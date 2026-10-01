"""
generate_gui.py -- a small desktop GUI for open-ended text generation from
one of language-cadence-2's trained checkpoints.

Defaults to runs/rope_seed1/checkpoint_best.pt: the best-test-loss seed of
the RoPE run, which was the strongest positional encoding in the sweep by a
wide margin (see writeup/paper.pdf, Section 4.1). Streams generated text
into a window indefinitely, one word at a time, from an optional start
string, until you click Stop.

This is a toy for poking at the trained model -- it isn't part of the
sweep/training pipeline. The model-loading/sampling/decoding logic lives in
src/generate.py (and is unit-tested there, see tests/test_generate.py);
this file is just the tkinter window around it. Run it from the repo root,
same convention as the other scripts/*.py:

    python3 scripts/generate_gui.py
    python3 scripts/generate_gui.py --checkpoint runs/rope_seed0/checkpoint_best.pt
    python3 scripts/generate_gui.py --device mps

Uses tkinter (Python's standard-library GUI toolkit), so no new dependency
is needed beyond what's already in requirements.txt. If `import tkinter`
fails: on macOS with Homebrew Python, `brew install python-tk@3.x` (match
your python3 version); on Debian/Ubuntu, `sudo apt install python3-tk`. The
python.org installer for macOS/Windows already bundles it, as does conda.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from generate import load_model, sample_next, IncrementalDecoder  # noqa: E402
from model import GPT, count_params  # noqa: E402
from syllable_tokenizer.vocab import Vocab, EOS  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join("runs", "rope_seed1", "checkpoint_best.pt")


class GeneratorApp:
    def __init__(self, root: tk.Tk, model: GPT, vocab: Vocab, cfg: dict, info: dict,
                 device: torch.device, run_id: str):
        self.root = root
        self.model = model
        self.vocab = vocab
        self.device = device

        self.stop_event = threading.Event()
        self.gen_thread: threading.Thread | None = None
        self.text_queue: "queue.Queue[str]" = queue.Queue()

        root.title(f"Language Cadence 2 -- generator ({run_id}, RoPE, best test-loss seed)")
        root.geometry("780x610")

        about = ttk.Frame(root, padding=(10, 10, 10, 0))
        about.pack(fill="x")
        pe_desc = {"rope": "rotary positional encoding (RoPE)",
                   "baseline": "a standard sinusoidal positional encoding, no periodic component",
                   "periodic": f"a periodic positional encoding (period={cfg.get('period')})"}[cfg["pe_mode"]]
        loss_bit = f", val loss {info['val_loss']:.4f}" if info.get("val_loss") is not None else ""
        about_text = (
            f"A {count_params(model) / 1e6:.2f}M-parameter GPT trained from scratch on English "
            f"TED-talk transcripts, syllable-tokenized with this project's own hybrid tokenizer, "
            f"using {pe_desc} -- the strongest configuration in our positional-encoding sweep. "
            f"Checkpoint: {run_id}{loss_bit}. See writeup/paper.pdf for the full write-up; this "
            f"generator is just a toy for poking at the trained model, not a research result."
        )
        ttk.Label(about, text=about_text, wraplength=740, justify="left",
                  foreground="#555555", font=("Helvetica", 10)).pack(fill="x")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Start text (optional):").grid(row=0, column=0, sticky="w")
        self.start_entry = ttk.Entry(top, width=60)
        self.start_entry.grid(row=0, column=1, columnspan=5, sticky="we", padx=(6, 0))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Temperature:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.temp_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(top, from_=0.1, to=2.0, increment=0.1, textvariable=self.temp_var, width=6).grid(
            row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Top-k:").grid(row=1, column=2, sticky="w", padx=(16, 0), pady=(8, 0))
        self.top_k_var = tk.IntVar(value=40)
        ttk.Spinbox(top, from_=0, to=len(vocab), increment=5, textvariable=self.top_k_var, width=6).grid(
            row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(top, text="(0 = no cutoff)").grid(row=1, column=4, sticky="w", padx=(6, 0), pady=(8, 0))

        btns = ttk.Frame(root, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        # Plain tk.Button (not ttk.Button) here on purpose: on some
        # platforms/themes (notably macOS's "aqua" ttk theme, especially in
        # dark mode) ttk.Button ignores explicit foreground/background
        # colors and can render white text on a white background. A plain
        # tk.Button always honors the colors below, regardless of system
        # theme.
        button_kwargs = dict(
            fg="#111111", bg="#e6e6e6",
            activeforeground="#111111", activebackground="#d5d5d5",
            disabledforeground="#888888",
            highlightbackground="#e6e6e6",
            relief="raised", borderwidth=1, padx=10, pady=3,
        )
        self.start_btn = tk.Button(btns, text="Start", command=self.on_start, **button_kwargs)
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(btns, text="Stop", command=self.on_stop, state="disabled", **button_kwargs)
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.clear_btn = tk.Button(btns, text="Clear", command=self.on_clear, **button_kwargs)
        self.clear_btn.pack(side="left", padx=(6, 0))
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(btns, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        self.output = scrolledtext.ScrolledText(root, wrap="word", font=("Georgia", 13))
        self.output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.output.config(state="disabled")

        self.root.after(40, self.poll_queue)

    def on_start(self):
        if self.gen_thread and self.gen_thread.is_alive():
            return
        start_text = self.start_entry.get()
        temperature = self.temp_var.get()
        top_k = self.top_k_var.get()
        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Generating... (click Stop to end)")
        self.gen_thread = threading.Thread(
            target=self.generate_loop, args=(start_text, temperature, top_k), daemon=True,
        )
        self.gen_thread.start()

    def on_stop(self):
        self.stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopping...")

    def on_clear(self):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.config(state="disabled")

    def generate_loop(self, start_text: str, temperature: float, top_k: int):
        vocab = self.vocab
        decoder = IncrementalDecoder(vocab)

        if start_text.strip():
            context = vocab.encode(start_text)
            for tid in context:
                decoder.feed(tid)  # prime the word buffer so the seed isn't re-emitted
            self.text_queue.put(start_text.rstrip() + " ")
        else:
            # No seed text: start from <eos>, the same token that preceded
            # every talk during training, so the model begins from roughly
            # the document-initial distribution it actually learned.
            context = [vocab.piece_to_id[EOS]]

        n_generated = 0
        while not self.stop_event.is_set():
            next_id = sample_next(self.model, context, self.device, temperature, top_k)
            context.append(next_id)
            n_generated += 1
            word = decoder.feed(next_id)
            if word:
                self.text_queue.put(word)
            if n_generated % 500 == 0:
                # sample_next() only ever looks at the last seq_len tokens
                # anyway; trim so this list doesn't grow forever across a
                # long-running session.
                context = context[-self.model.seq_len:]

        tail = decoder.flush()
        if tail:
            self.text_queue.put(tail)
        self.text_queue.put("__STOPPED__")

    def poll_queue(self):
        try:
            while True:
                chunk = self.text_queue.get_nowait()
                if chunk == "__STOPPED__":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set("Stopped.")
                    continue
                self.output.config(state="normal")
                self.output.insert(tk.END, chunk)
                self.output.see(tk.END)
                self.output.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(40, self.poll_queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"Path to a trained checkpoint (default: {DEFAULT_CHECKPOINT}, "
                              f"the best-test-loss RoPE seed from the sweep).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                         help="Defaults to cuda > mps > cpu, whichever is available.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.checkpoint):
        parser.error(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Trained checkpoints aren't committed to the repo (see .gitignore) -- "
            f"run the sweep first (scripts/run_sweep.py), or point --checkpoint at "
            f"wherever your runs/ directory actually lives."
        )

    model, vocab, cfg, info = load_model(args.checkpoint, device)
    run_id = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))

    root = tk.Tk()
    GeneratorApp(root, model, vocab, cfg, info, device, run_id)
    root.mainloop()


if __name__ == "__main__":
    main()
