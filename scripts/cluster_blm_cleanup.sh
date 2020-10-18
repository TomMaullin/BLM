#!/bin/bash
fslpython -c "from src import blm_cleanup; blm_cleanup.main('$1')"
