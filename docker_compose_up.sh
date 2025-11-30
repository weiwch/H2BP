docker build . -t mganns_img

ssh-keygen -t rsa -f ./id_rsa -q -N ""

docker compose up -d