FROM sequenceiq/hadoop-docker:2.7.1

COPY ./src/docker/hadoop-config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY ./src/docker/hadoop-config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml

COPY ./src/data /data

USER root
RUN rm -rf /data/datanode && mkdir -p /data/datanode && mv /data/*.csv.bz2 /data/datanode/ && \
    mkdir -p /data/namenode

CMD ["/etc/bootstrap.sh", "-d"]
