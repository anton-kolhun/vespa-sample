package ai.vespa.example;


import ai.vespa.models.evaluation.ModelsEvaluator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.Field;
import com.yahoo.document.TensorDataType;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.document.update.AssignValueUpdate;
import com.yahoo.document.update.FieldUpdate;
import com.yahoo.tensor.Tensor;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

public class ImageEmbedderProcessor extends DocumentProcessor {

    private final ModelsEvaluator modelsEvaluator;
    private final HttpClient httpClient;
    private final URI mathEngineUri;
    private final ImageEmbedderConfig config;
    private final Map<String, FieldConfig> schemaToFieldConfig = new HashMap<>();

    @Inject
    public ImageEmbedderProcessor(ModelsEvaluator modelsEvaluator, ImageEmbedderConfig cfg) {
        this.modelsEvaluator = modelsEvaluator;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        this.mathEngineUri = URI.create("http://math-engine:8001/transform_to_tensor");
        this.config = cfg;
        cfg.schemaToFieldsCfg().forEach((schema, fields) -> {
            String[] fieldSplit = fields.split(",");
            schemaToFieldConfig.put(schema, new FieldConfig(fieldSplit[0].trim(), fieldSplit[1].trim()));
        });
    }

    static Logger log = Logger.getLogger(ImageEmbedderProcessor.class.getName());

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            try {
                if (op instanceof DocumentUpdate) {
                    DocumentUpdate docUpdate = (DocumentUpdate) op;
                    String documentType = docUpdate.getDocumentType().getName();
                    if (!schemaToFieldConfig.containsKey(documentType)) {
                        continue;
                    }
                    FieldConfig fieldConfig = schemaToFieldConfig.get(documentType);
                    AssignValueUpdate valueUpdate = (AssignValueUpdate) docUpdate.getFieldUpdate(fieldConfig.fromField).getValueUpdate(0);
                    String imageUrl = valueUpdate.getValue().toString();
                    String embField = fieldConfig.toField;
                    Tensor embedding = calculateEmbedding(imageUrl);
                    FieldUpdate embFieldUpdate = FieldUpdate.createAssign(
                            new Field(embField, new TensorDataType(embedding.type())), new TensorFieldValue(embedding));
                    docUpdate.addFieldUpdate(embFieldUpdate);
                } else if (op instanceof DocumentPut) {
                    DocumentPut put = (DocumentPut) op;
                    Document document = put.getDocument();
                    if (!schemaToFieldConfig.containsKey(document.getDataType().getName())) {
                        continue;
                    }
                    FieldConfig fieldConfig = schemaToFieldConfig.get(document.getDataType().getName());
                    String imageFileName = document.getFieldValue(fieldConfig.fromField).toString();
                    Tensor embedding = calculateEmbedding(imageFileName);
                    document.setFieldValue(fieldConfig.toField, new TensorFieldValue(embedding));
                }
            } catch (Exception e) {
                log.warning("Error occurred while calculating image embedding. Suppressing and continuing the execution..." + e);
            }
        }
        return Progress.DONE;
    }

    private Tensor calculateEmbedding(String imageUrl) throws IOException, InterruptedException {
        Map<String, String> payload = Map.of("image_url", imageUrl);
        var om = new ObjectMapper();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(mathEngineUri)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(om.writeValueAsString(payload)))
                .timeout(Duration.ofMillis(5000))
                .build();
        String result = httpClient.send(request, HttpResponse.BodyHandlers.ofString()).body();
        Tensor tensor = Tensor.from("tensor<float>(d0[1],d1[3], d2[224], d3[224])", result);
        Tensor embedding = modelsEvaluator.evaluatorOf(config.modelName()).bind("input_t", tensor).evaluate();
        embedding = Util.slice(embedding, "d0:0").rename("d1", "x").l2Normalize("x");
        return embedding;
    }

    @Override
    public void deconstruct() {
        modelsEvaluator.deconstruct();
    }

    private static class FieldConfig {

        private final String fromField;
        private final String toField;

        public FieldConfig(String fromField, String toField) {
            this.fromField = fromField;
            this.toField = toField;
        }
    }
}