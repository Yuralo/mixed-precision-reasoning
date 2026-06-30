# Research report

Serve the repository root so the report can display figures from `runs/`:

```bash
python -m http.server 8000
```

Open <http://localhost:8000/docs/>.

The report is printable from the browser and includes print-specific styling. Figure
cards automatically use the PNG files created by:

```bash
python -m scripts.make_figures
```

Until a figure exists, its card displays a clear placeholder.

Supporting documents:

- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md): complete project explanation from first principles.
- [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md): Stage 1 statistics, limitations, novelty review, and next experiments.
- [`TOKEN_LENGTH_STUDY.md`](TOKEN_LENGTH_STUDY.md): paired study of quantization, output length, rescues, and failures.
