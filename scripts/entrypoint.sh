cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
service ssh start
tail -f /dev/null

echo "export LD_LIBRARY_PATH=/anns/thirdparty/faiss/_libfaiss_stage/lib:$LD_LIBRARY_PATH" >> ~/.bashrc