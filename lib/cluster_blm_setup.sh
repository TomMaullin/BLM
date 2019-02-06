#!/bin/bash
fslpython -c "from BLM import blm_setup; blm_setup.main('$1')"
