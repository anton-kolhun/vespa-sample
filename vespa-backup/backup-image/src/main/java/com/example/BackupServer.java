package com.example;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;


public class BackupServer {

    public static void main(String[] args) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8000), 0);
        server.createContext("/", new BackupHandler());
        server.start();
    }

    static class BackupHandler implements HttpHandler {

        private static final ExecutorService es = Executors.newFixedThreadPool(4);

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod()) || !exchange.getRequestURI().getPath().equals("/backup")) {
                respond(exchange, "HTTP method or URI not supported", 500);
                return;
            }
            triggerBackup(exchange);
        }

        private void triggerBackup(HttpExchange httpExchange) throws IOException {
            Map<String, String> params = queryToMap(httpExchange.getRequestURI().getQuery());
            int chunks = Integer.parseInt(params.getOrDefault("chunks", "10"));
            int threads = Integer.parseInt(params.getOrDefault("threads", "10"));
            Calendar startDate = Calendar.getInstance();
            int finalChunks = chunks, finalThreads = threads;
            CompletableFuture.runAsync(() -> Processor.run(finalChunks, finalThreads, startDate), es);
            respond(httpExchange, "Backup started. Check log files...", 200);
        }

        private static void respond(HttpExchange exchange, String msg, int code) throws IOException {
            try (OutputStream out = exchange.getResponseBody()) {
                byte[] body = msg.getBytes();
                exchange.sendResponseHeaders(code, body.length);
                out.write(body);
            }
        }

        private static Map<String, String> queryToMap(String query) {
            if (query == null) return Collections.emptyMap();
            Map<String, String> result = new HashMap<>();
            for (String param : query.split("&")) {
                String[] entry = param.split("=");
                result.put(URLDecoder.decode(entry[0]), entry.length > 1 ? URLDecoder.decode(entry[1]) : "");
            }
            return result;
        }
    }

    static class Processor {

        private static final Logger log = Logger.getLogger(Processor.class.getName());
        private static final String DEFAULT_CONTENT_CLUSTER = "content";

        private static final ExecutorService compressAndUploadExecutor = Executors.newFixedThreadPool(10);
        private static final String s3BasePath = String.format("%s", System.getenv().getOrDefault("VESPA_BACKUP_S3_BUCKET", "s3://your-vespa-backup-bucket/"));

        static void run(int chunks, int threads, Calendar startDate) {
            String cluster = System.getenv().getOrDefault("CONTENT_CLUSTER", DEFAULT_CONTENT_CLUSTER);
            ExecutorService visitExecutor = Executors.newFixedThreadPool(threads);
            List<Future<String>> uploads = new ArrayList<>();
            List<CompletableFuture<String>> visits = new ArrayList<>();
            for (int i = 0; i < chunks; i++) {
                int sliceId = i;
                CompletableFuture<String> visit = CompletableFuture.supplyAsync(() -> doVisit(sliceId, chunks, cluster), visitExecutor);
                visit = visit.whenComplete((file, err) -> {
                    if (err == null) uploads.add(compressAndUploadExecutor.submit(() -> {
                        String gz = gzip(file);
                        upload(gz, startDate);
                        remove(gz);
                        return file;
                    }));
                });
                visits.add(visit);
            }
            visits.forEach(f -> { try { f.get(); } catch (Exception e) { log.severe(e.toString()); } });
            uploads.forEach(f -> { try { f.get(); } catch (Exception e) { log.severe(e.toString()); } });
            log.info("Backup completed");
        }

        private static String doVisit(int sliceId, int chunks, String cluster) {
            String dumpFile = String.format("backup_%s.json", sliceId);
            File logFile = new File(String.format("backup_%s.log", sliceId));
            String cmd = String.format(
                    "vespa-visit --cluster %s --slices %s --sliceid %s --progress backup_%s.progress" +
                    " --shorttensors > %s", cluster, chunks, sliceId, sliceId, dumpFile);
            try {
                int exit = new ProcessBuilder("/bin/sh", "-c", cmd).redirectOutput(logFile).redirectError(logFile).start().waitFor();
                if (exit == 0) { log.info("visit done: " + dumpFile); return dumpFile; }
            } catch (Exception e) {
                throw new RuntimeException("vespa-visit failed, see " + logFile.getName(), e);
            }
            throw new RuntimeException("vespa-visit failed, see " + logFile.getName());
        }

        private static String gzip(String file) throws IOException, InterruptedException {
            File logFile = new File(file + "_compression.log");
            int exit = new ProcessBuilder("/bin/sh", "-c", "gzip -f " + file).redirectOutput(logFile).redirectError(logFile).start().waitFor();
            if (exit == 0) return file + ".gz";
            throw new RuntimeException("gzip failed, see " + logFile.getName());
        }

        private static void upload(String file, Calendar date) throws IOException, InterruptedException {
            String s3Path = s3BasePath + "/" + date.get(Calendar.YEAR) + "/" + (date.get(Calendar.MONTH) + 1)
                    + "/" + date.get(Calendar.DAY_OF_MONTH) + "/" + file;
            File logFile = new File(file + "_s3.log");
            int exit = new ProcessBuilder("/bin/sh", "-c", "aws s3 cp " + file + " " + s3Path)
                    .redirectOutput(logFile).redirectError(logFile).start().waitFor();
            if (exit != 0) throw new RuntimeException("s3 upload failed, see " + logFile.getName());
        }

        private static void remove(String file) throws IOException, InterruptedException {
            File logFile = new File(file + "_deletion.log");
            int exit = new ProcessBuilder("/bin/sh", "-c", "rm -f " + file)
                    .redirectOutput(logFile).redirectError(logFile).start().waitFor();
            if (exit != 0) throw new RuntimeException("delete failed, see " + logFile.getName());
        }

    }
}
