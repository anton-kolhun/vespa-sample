# Stage 1 — build the backup handler JAR
FROM maven:3.9.9-amazoncorretto-17-debian AS builder
COPY src /app/src
COPY pom.xml /app/pom.xml
WORKDIR /app
RUN mvn package -q

# Stage 2 — extend the official Vespa image
FROM vespaengine/vespa:8.738.17

USER root
RUN yum install -y awscli

COPY --from=builder /app/target/vespa-backup-handler-1.0-SNAPSHOT-jar-with-dependencies.jar \
     /opt/vespa-backup-handler.jar
COPY restore_backup.sh /opt/backup/restore_backup.sh

RUN chown vespa /opt/vespa-backup-handler.jar /opt/backup /opt/backup/restore_backup.sh && \
    chmod +x /opt/backup/restore_backup.sh

USER vespa
WORKDIR /opt/backup

ENTRYPOINT ["bash", "-c", \
  "/opt/vespa/bin/vespa-start-services && \
   java -cp /opt/vespa-backup-handler.jar:/opt/vespa/lib/jars/* com.example.BackupServer"]
