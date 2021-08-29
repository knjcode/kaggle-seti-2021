#!/bin/bash

host_file=$1
worker_num=$2
options=${@:3:($#-2)}

LC_ALL=C mpiexec \
--allow-run-as-root \
--mca pml ob1 \
--mca orte_base_help_aggregate 0 \
--mca btl ^openib \
--mca btl_tcp_if_include ib0 \
-x PATH=$PATH \
-x PYTHONPATH=$PYTHONOATH \
-x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
-x CPATH=$CPATH \
-x LIBRARY_PATH=$LIBRARY_PATH \
-x NCCL_ROOT=$NCCL_ROOT \
-x NCCL_DEBUG=INFO \
-bind-to none \
-map-by slot \
--hostfile ${host_file} \
-np ${worker_num} \
/opt/conda/bin/python -u ddp_horovod_train.py ${options}
