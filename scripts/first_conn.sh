NODE_COUNT=$1

for i in $(seq 1 $NODE_COUNT)
do
  HOSTNAME="node${i}"
  ssh -o "StrictHostKeyChecking=accept-new" ${HOSTNAME} 'exit'
done