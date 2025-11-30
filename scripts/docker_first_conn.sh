NODE_COUNT=$1

for i in $(seq 1 $NODE_COUNT)
do
  HOSTNAME="node${i}"
  docker exec $HOSTNAME bash /anns/scripts/first_conn.sh $NODE_COUNT
done