python_path=$(which python)

jman submit -n VERAFinger-wld -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_wld.py

jman submit -n VERAFinger-mc -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_mc.py

jman submit -n VERAFinger-rlt -q  lgpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_rlt.py
