python_path=$(which python)
jman submit -n wld -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path wld.py 
jman submit -n mc  -q  gpu --environment "PYTHONUNBUFFERED=1" -- $python_path mc.py 
jman submit -n rlt -q lgpu --environment "PYTHONUNBUFFERED=1" -- $python_path RLT.py

jman list