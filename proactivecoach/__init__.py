"""ProactiveCoach sparse, two-second streaming SFT reference implementation."""
import os

# Fix the selected recipe before importing its prompt/video helpers.
for key, value in {
    'THINKSTREAM_PROMPT': 'SPARSE_GUIDE',
    'THINKSTREAM_CHUNK_SIZE': '2',
    'THINKSTREAM_SPARSE_PRESERVE_LEGACY': '1',
    'THINKSTREAM_VIDEO_BACKEND': 'av',
    'FORCE_QWENVL_VIDEO_READER': 'torchvision',
    'THINKSTREAM_Q35_WINDOW': '1',
    'THINKSTREAM_DISABLE_RESPONSE_VIDEO_MASK': '1',
}.items():
    os.environ[key] = value
