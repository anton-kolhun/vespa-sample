// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import javax.inject.Inject;

public class TextEmbeddingSearcher extends Searcher {

    private static final String TEXT_EMBEDDING_ENABLED_FIELD = "text_embedding_enabled";

    private final ModelsEvaluator modelsEvaluator;
    private final BPETokenizer tokenizer;
    private final TextEmbedderConfig config;

    @Inject
    public TextEmbeddingSearcher(ModelsEvaluator modelsEvaluator, BPETokenizer tokenizer,
                                 TextEmbedderConfig cfg) {
        this.modelsEvaluator = modelsEvaluator;
        this.tokenizer = tokenizer;
        this.config = cfg;
    }

    @Override
    public Result search(Query query, Execution execution) {
        boolean textEmbeddingEnabled = query.properties().getBoolean(TEXT_EMBEDDING_ENABLED_FIELD);
        if (!textEmbeddingEnabled) {
            return execution.search(query);
        }
        String queryString = query.properties().getString("search_term", null);
        if (queryString == null || queryString.isBlank()) {
            return new Result(query, ErrorMessage.createBadRequest("No 'search_term' query param"));
        }
        Tensor input = tokenizer.encode(queryString).rename("d0", "d1").expand("d0");
        Tensor embedding = modelsEvaluator.evaluatorOf(config.modelName()).bind("input", input).evaluate();
        embedding = Util.slice(embedding, "d0:0").rename("d1", "x").l2Normalize("x");
        query.getRanking().getFeatures().put(config.rankFeatureParam(), embedding);
        return execution.search(query);
    }

    @Override
    public void deconstruct() {
        modelsEvaluator.deconstruct();
    }

}

