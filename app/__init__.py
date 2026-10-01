# On macOS, faiss and torch each bundle their own OpenMP runtime. If faiss is imported first,
# loading a large model with torch afterwards (the local chat model) segfaults. Importing torch
# before anything in app/ touches faiss avoids it. torch is optional (only the local re-ranker
# and local models need it), so this is skipped when it isn't installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass
