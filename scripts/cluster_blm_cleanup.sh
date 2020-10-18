#!/bin/bash
fslpython -c "from lib import blm_cleanup; blm_cleanup.main('$1')"
