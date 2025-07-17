#!/bin/bash
set -x

# USE CUDA TRUE
# USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration # don't waste time on full + fake checkpoint
USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -s
USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -s
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -s  # should error 
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration -s # should error
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -s
USE_CUDA=True USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -s

# USE CUDA FALSE
# USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration # don't waste time on full + fake checkpoint
USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -s
USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -s
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -s # should error
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=True USE_REAL_CHECKPOINTS=False pytest -m integration -s # should error
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=True pytest -m integration -s
USE_CUDA=False USE_REAL_DATA=False USE_FULL_DATA=False USE_REAL_CHECKPOINTS=False pytest -m integration -s


# EXTRA EXPENSIVE ONES

USE_CUDA=True USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -s  # expensive

USE_CUDA=False USE_REAL_DATA=True USE_FULL_DATA=True USE_REAL_CHECKPOINTS=True pytest -m integration -s # very expensive
