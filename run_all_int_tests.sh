#!/bin/bash
set -x

# USE CUDA TRUE
# USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration # don't waste time on full + fake checkpoint
USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -rAV
USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -rAV
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -rAV  # should error
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration -rAV # should error
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -rAV
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -rAV

# USE CUDA FALSE
# USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration # don't waste time on full + fake checkpoint
USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -rAV
USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -rAV
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -rAV # should error
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration -rAV # should error
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -rAV
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -rAV


# EXTRA EXPENSIVE ONES

USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -rAV  # expensive

USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -rAV # very expensive
