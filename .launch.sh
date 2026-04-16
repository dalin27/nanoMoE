runai submit \
  --name transfer-pod \
  --image nvcr.io/nvidia/pytorch:25.10-py3 \
  --pvc course-ee-628-scratch:/scratch \
  --interactive \
  --gpu 0 \
  -- bash
