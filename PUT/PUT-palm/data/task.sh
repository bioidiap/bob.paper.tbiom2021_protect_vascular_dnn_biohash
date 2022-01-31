python_path=$(which python)

#jman submit -n wrist-feats -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats.py

jman submit -n wrist-wld -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_wld.py

jman submit -n wrist-mc -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_mc.py

jman submit -n wrist-rlt -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path feats_rlt.py