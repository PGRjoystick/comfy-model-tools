"""Windows-friendly GUI for quant_int8_convrot.py (int8 / W6A8 / W4A8 ConvRot quantization).

    python convrot_gui.py                 # open the GUI
    python convrot_gui.py --selftest LOG  # headless check of the bundled converter (used by the CI build)

Packaged as a portable folder with PyInstaller (see convrot_gui.spec). The converter runs in a
child process so the window stays responsive, Cancel can stop it at any point, and all GPU/CPU
memory is released when a job ends. torch is only imported in that child, so the window opens fast.
"""
import multiprocessing
import os
import queue
import re
import sys
import time
import traceback

APP_NAME = "ConvRot Quantizer"
MODEL_TYPES = [("Model files", "*.safetensors *.pth *.pt *.ckpt *.bin"), ("All files", "*.*")]
FORMATS = {   # label -> (extra args, output tag)
    "INT8 (int8 + ConvRot)": ([], "int8_convrot"),
    "W6A8 (int6 weights, ~0.78 B/weight)": (["--w6a8"], "w6a8_convrot"),
    "W4A8 (int4 weights + codebook, ~0.56 B/weight)": (["--w4a8"], "w4a8_convrot"),
}


def default_dst(src, tag):
    """Same rule as the CLI: swap bf16/fp16/fp32 in the name for the tag, else append it."""
    base = os.path.splitext(os.path.basename(src))[0]
    new = re.sub(r"(?i)(bf16|fp16|fp32)", tag, base)
    if new == base:
        new = base + "_" + tag
    return os.path.join(os.path.dirname(src), new + ".safetensors")


# ---------------------------------------------------------------------------
# child process: run the converter, stream its output back line by line
# ---------------------------------------------------------------------------
class _QueueWriter:
    def __init__(self, q):
        self.q, self.buf = q, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(("line", line))
        return len(s)

    def flush(self):
        if self.buf:
            self.q.put(("line", self.buf))
            self.buf = ""

    def isatty(self):
        return False


def _worker(argv, q):
    sys.stdout = sys.stderr = _QueueWriter(q)   # windowed exe has no console: capture everything
    code = 0
    try:
        import torch
        import quant_int8_convrot
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})")
        else:
            print(f"No CUDA GPU available, running on CPU (torch {torch.__version__})")
        quant_int8_convrot.main(argv)
    except SystemExit as e:                      # argparse errors
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    q.put(("exit", code))


class Job:
    """One converter run in a child process."""

    def __init__(self, argv):
        ctx = multiprocessing.get_context("spawn")    # same behaviour on every OS, never fork tk
        self.q = ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(argv, self.q), daemon=True)
        self.proc.start()
        self.exit_code = None

    def poll(self):
        """Return the output lines produced so far; sets exit_code once the run is over."""
        lines = []
        alive = self.proc.is_alive()     # check first: a child that exits mid-drain is still drained
        try:
            while True:
                kind, val = self.q.get(timeout=0.5) if not alive else self.q.get_nowait()
                if kind == "line":
                    lines.append(val)
                else:
                    self.exit_code = val
        except queue.Empty:
            pass
        if self.exit_code is None and not alive:
            self.exit_code = self.proc.exitcode if self.proc.exitcode else 1
            lines.append(f"converter process died unexpectedly (exit code {self.proc.exitcode}) "
                         "- out of RAM?")
        return lines

    def cancel(self):
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title(APP_NAME)
    root.minsize(760, 560)
    pad = {"padx": 6, "pady": 3}

    src_var, dst_var = tk.StringVar(), tk.StringVar()
    fmt_var = tk.StringVar(value=next(iter(FORMATS)))
    device_var = tk.StringVar(value="auto")
    mseclip_var, downcast_var, embed_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(value=True)
    min_gemm_var, group_var, warn_var = tk.StringVar(value="256"), tk.StringVar(), tk.StringVar()
    exclude_var, include_var, report_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    status_var = tk.StringVar(value="Pick a source model.")
    state = {"job": None, "dst_auto": True, "dst": None, "dst_mtime": None, "t0": 0.0}

    def auto_dst():
        if src_var.get() and state["dst_auto"]:
            dst_var.set(default_dst(src_var.get(), FORMATS[fmt_var.get()][1]))

    def browse_src():
        p = filedialog.askopenfilename(title="Source model", filetypes=MODEL_TYPES)
        if p:
            src_var.set(os.path.normpath(p))
            auto_dst()
            status_var.set("Ready. Dry run shows what will be quantized without writing anything.")

    def browse_dst():
        init = dst_var.get() or (src_var.get() and default_dst(src_var.get(), FORMATS[fmt_var.get()][1]))
        p = filedialog.asksaveasfilename(
            title="Save quantized model as", defaultextension=".safetensors",
            initialdir=os.path.dirname(init) if init else None,
            initialfile=os.path.basename(init) if init else None,
            filetypes=[("safetensors", "*.safetensors")])
        if p:
            dst_var.set(os.path.normpath(p))
            state["dst_auto"] = False

    def browse_report():
        p = filedialog.asksaveasfilename(title="Per-layer error report", defaultextension=".tsv",
                                         filetypes=[("TSV", "*.tsv"), ("Text", "*.txt")])
        if p:
            report_var.set(os.path.normpath(p))

    # --- files ---
    files = ttk.LabelFrame(root, text="Files")
    files.pack(fill="x", padx=8, pady=(8, 4))
    files.columnconfigure(1, weight=1)
    ttk.Label(files, text="Source model:").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(files, textvariable=src_var).grid(row=0, column=1, sticky="ew", **pad)
    ttk.Button(files, text="Browse...", command=browse_src).grid(row=0, column=2, **pad)
    ttk.Label(files, text="Output:").grid(row=1, column=0, sticky="w", **pad)
    dst_entry = ttk.Entry(files, textvariable=dst_var)
    dst_entry.grid(row=1, column=1, sticky="ew", **pad)
    dst_entry.bind("<Key>", lambda e: state.update(dst_auto=False))
    ttk.Button(files, text="Browse...", command=browse_dst).grid(row=1, column=2, **pad)
    ttk.Label(files, text="Source: .safetensors, or .pth/.pt/.ckpt/.bin (weights-only load). "
                          "Output is always .safetensors.", foreground="gray").grid(
        row=2, column=0, columnspan=3, sticky="w", **pad)

    # --- options ---
    opts = ttk.LabelFrame(root, text="Options")
    opts.pack(fill="x", padx=8, pady=4)
    ttk.Label(opts, text="Format:").grid(row=0, column=0, sticky="w", **pad)
    fmt_box = ttk.Combobox(opts, textvariable=fmt_var, values=list(FORMATS), state="readonly", width=46)
    fmt_box.grid(row=0, column=1, columnspan=3, sticky="w", **pad)
    fmt_box.bind("<<ComboboxSelected>>", lambda e: auto_dst())
    ttk.Label(opts, text="Device:").grid(row=0, column=4, sticky="e", **pad)
    ttk.Combobox(opts, textvariable=device_var, values=["auto", "cuda", "cpu"], width=8).grid(
        row=0, column=5, sticky="w", **pad)
    ttk.Checkbutton(opts, text="Quantize token embeddings", variable=embed_var).grid(
        row=1, column=0, columnspan=2, sticky="w", **pad)
    ttk.Checkbutton(opts, text="MSE-optimal clip (experimental, int8 only)", variable=mseclip_var).grid(
        row=1, column=2, columnspan=2, sticky="w", **pad)
    ttk.Checkbutton(opts, text="Downcast stray fp32", variable=downcast_var).grid(
        row=1, column=4, columnspan=2, sticky="w", **pad)

    adv = ttk.LabelFrame(root, text="Advanced (leave empty for defaults)")
    adv.pack(fill="x", padx=8, pady=4)
    adv.columnconfigure(1, weight=1)
    adv.columnconfigure(3, weight=1)
    ttk.Label(adv, text="Min GEMM:").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(adv, textvariable=min_gemm_var, width=8).grid(row=0, column=1, sticky="w", **pad)
    ttk.Label(adv, text="Group size (W4A8/W6A8):").grid(row=0, column=2, sticky="w", **pad)
    ttk.Entry(adv, textvariable=group_var, width=8).grid(row=0, column=3, sticky="w", **pad)
    ttk.Label(adv, text="Warn threshold %:").grid(row=0, column=4, sticky="w", **pad)
    ttk.Entry(adv, textvariable=warn_var, width=8).grid(row=0, column=5, sticky="w", **pad)
    ttk.Label(adv, text="Exclude regex:").grid(row=1, column=0, sticky="w", **pad)
    ttk.Entry(adv, textvariable=exclude_var).grid(row=1, column=1, columnspan=5, sticky="ew", **pad)
    ttk.Label(adv, text="Include regex:").grid(row=2, column=0, sticky="w", **pad)
    ttk.Entry(adv, textvariable=include_var).grid(row=2, column=1, columnspan=5, sticky="ew", **pad)
    ttk.Label(adv, text="Error report file:").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(adv, textvariable=report_var).grid(row=3, column=1, columnspan=4, sticky="ew", **pad)
    ttk.Button(adv, text="Browse...", command=browse_report).grid(row=3, column=5, sticky="w", **pad)

    # --- actions + log ---
    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=8, pady=4)
    dry_btn = ttk.Button(bar, text="Dry run (plan only)")
    run_btn = ttk.Button(bar, text="Convert")
    cancel_btn = ttk.Button(bar, text="Cancel", state="disabled")
    dry_btn.pack(side="left", padx=(0, 6))
    run_btn.pack(side="left", padx=(0, 6))
    cancel_btn.pack(side="left")
    progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
    progress.pack(side="right")

    log = ScrolledText(root, height=16, wrap="none", font=("Consolas", 9))
    log.pack(fill="both", expand=True, padx=8, pady=4)
    log.tag_configure("warn", foreground="#b36b00")
    log.tag_configure("err", foreground="#c00000")
    log.tag_configure("ok", foreground="#007a00")
    ttk.Label(root, textvariable=status_var, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

    def append(line, tag=None):
        if tag is None:
            low = line.lower()
            if "warn" in low or line.lstrip().startswith("!!"):
                tag = "warn"
            elif re.search(r"Traceback|\w+(Error|Exception)\b|^\s+File \"|died unexpectedly", line):
                tag = "err"
            elif line.startswith("DONE"):
                tag = "ok"
        log.insert("end", line + "\n", tag or ())
        log.see("end")

    def build_argv(dry):
        src = src_var.get().strip().strip('"')
        if not src or not os.path.isfile(src):
            messagebox.showerror(APP_NAME, "Pick an existing source model file first.")
            return None
        argv = [src]
        if not dry:
            dst = dst_var.get().strip().strip('"') or default_dst(src, FORMATS[fmt_var.get()][1])
            if os.path.abspath(dst) == os.path.abspath(src):
                messagebox.showerror(APP_NAME, "Output must be different from the source file.")
                return None
            if os.path.exists(dst) and not messagebox.askyesno(
                    APP_NAME, f"{dst}\n\nalready exists. Overwrite it?"):
                return None
            argv.append(dst)
            state["dst"] = dst
            state["dst_mtime"] = os.path.getmtime(dst) if os.path.exists(dst) else None
        else:
            argv.append("--dry-run")
            state["dst"] = None
        argv += FORMATS[fmt_var.get()][0]
        argv += ["--device", device_var.get().strip() or "auto"]
        for flag, var in (("--min-gemm", min_gemm_var), ("--group-size", group_var),
                          ("--warn-thresh", warn_var)):
            v = var.get().strip()
            if v:
                try:
                    float(v) if flag == "--warn-thresh" else int(v)
                except ValueError:
                    messagebox.showerror(APP_NAME, f"{flag[2:]} must be a number, got {v!r}.")
                    return None
                argv += [flag, v]
        for flag, var in (("--exclude", exclude_var), ("--include", include_var)):
            v = var.get().strip()
            if v:
                try:
                    re.compile(v)
                except re.error as e:
                    messagebox.showerror(APP_NAME, f"Invalid {flag[2:]} regex: {e}")
                    return None
                argv += [flag, v]
        if report_var.get().strip() and not dry:
            argv += ["--verify-report", report_var.get().strip()]
        if mseclip_var.get():
            argv.append("--mseclip")
        if downcast_var.get():
            argv.append("--downcast-fp32")
        if not embed_var.get():
            argv.append("--no-quant-embeddings")
        return argv

    def set_running(running):
        for b in (dry_btn, run_btn):
            b.configure(state="disabled" if running else "normal")
        cancel_btn.configure(state="normal" if running else "disabled")
        (progress.start if running else progress.stop)(12)

    def start(dry):
        if state["job"]:
            return
        argv = build_argv(dry)
        if argv is None:
            return
        log.delete("1.0", "end")
        append("> quant_int8_convrot " + " ".join(f'"{a}"' if " " in a else a for a in argv), "ok")
        status_var.set("Dry run..." if dry else "Converting... (loading torch, this can take a moment)")
        state["t0"] = time.time()
        state["job"] = Job(argv)
        set_running(True)
        root.after(100, pump)

    def pump():
        job = state["job"]
        if job is None:
            return
        for line in job.poll():
            append(line)
        if job.exit_code is None:
            root.after(100, pump)
            return
        state["job"] = None
        set_running(False)
        mins = (time.time() - state["t0"]) / 60
        if job.exit_code == 0:
            status_var.set(f"Finished in {mins:.1f} min." + (f"  Output: {state['dst']}" if state["dst"] else ""))
        else:
            status_var.set(f"Failed (exit code {job.exit_code}) - see the log above.")

    def cancel():
        job = state["job"]
        if not job:
            return
        job.cancel()
        state["job"] = None
        set_running(False)
        append("Cancelled.", "warn")
        dst = state["dst"]
        if dst and os.path.exists(dst) and os.path.getmtime(dst) != state["dst_mtime"]:
            try:
                os.remove(dst)               # partially written output
                append(f"Removed incomplete output {dst}", "warn")
            except OSError as e:
                append(f"Could not remove incomplete output {dst}: {e}", "err")
        status_var.set("Cancelled.")

    def on_close():
        if state["job"]:
            if not messagebox.askyesno(APP_NAME, "A conversion is running. Cancel it and quit?"):
                return
            cancel()
        root.destroy()

    dry_btn.configure(command=lambda: start(True))
    run_btn.configure(command=lambda: start(False))
    cancel_btn.configure(command=cancel)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ---------------------------------------------------------------------------
# headless self-test: tiny synthetic model through every format via the same child-process path
# ---------------------------------------------------------------------------
def selftest(log_path):
    out = open(log_path, "w", encoding="utf-8")

    def say(s):
        out.write(s + "\n")
        out.flush()

    ok = True
    try:
        ok = _selftest_runs(say)
    except BaseException:                        # never let a windowed exe die with a modal dialog
        say(traceback.format_exc())
        ok = False
    say("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    out.close()
    return 0 if ok else 1


def _selftest_runs(say):
    import tempfile
    ok = True
    # ignore_cleanup_errors: on Windows a still-mapped output file can't be deleted right away
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        src = os.path.join(d, "tiny_bf16.safetensors")
        say("creating synthetic model")
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
        g = torch.Generator().manual_seed(0)
        sd = {}
        for i in range(2):
            for name, shape in (("attn.qkv", (768, 256)), ("attn.proj", (256, 256)),
                                ("mlp.fc1", (512, 256)), ("mlp.fc2", (256, 512))):
                sd[f"blocks.{i}.{name}.weight"] = (torch.randn(shape, generator=g) * 0.02).bfloat16()
                sd[f"blocks.{i}.{name}.bias"] = torch.zeros(shape[0], dtype=torch.bfloat16)
            sd[f"blocks.{i}.norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
        sd["embed_tokens.weight"] = (torch.randn((8192, 256), generator=g) * 0.02).bfloat16()
        sd["final_layer.weight"] = (torch.randn((64, 256), generator=g) * 0.02).bfloat16()
        save_file(sd, src)

        runs = [("dry-run", ["--dry-run"], None)]
        for label, (extra, tag) in FORMATS.items():
            runs.append((label, extra, default_dst(src, tag)))
        for label, extra, dst in runs:
            argv = [src] + ([dst] if dst else []) + extra + ["--device", "cpu"]
            say(f"\n=== {label}: {' '.join(argv)}")
            job, t0 = Job(argv), time.time()
            while job.exit_code is None:
                for line in job.poll():
                    say(line)
                if time.time() - t0 > 300:
                    job.cancel()
                    say("TIMEOUT: converter child did not finish in 300s")
                    return False
                time.sleep(0.05)
            for line in job.poll():
                say(line)
            good = job.exit_code == 0
            if good and dst:
                with safe_open(dst, "pt") as f:
                    keys = set(f.keys())
                good = "blocks.0.attn.qkv.comfy_quant" in keys and "embed_tokens.comfy_quant" in keys
            say(f"--- {label}: {'PASS' if good else 'FAIL'} (exit {job.exit_code})")
            ok &= good
    return ok


if __name__ == "__main__":
    multiprocessing.freeze_support()             # required for the child process in a frozen exe
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    run_gui()
