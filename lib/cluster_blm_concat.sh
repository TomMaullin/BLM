#!/bin/bash
fslpython -c "from BLM import blm_concat; blm_concat.main($1)"
