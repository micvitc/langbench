import pytest
import pandas as pd
from langbench.metrics import ToxicityMetric, BiasMetric


# Fake pipeline for ToxicityMetric
def fake_pipeline_toxic(text, top_k=None):
    if text == "toxic text":
        return [{"score": 0.95, "label": "toxic"}]
    # For non-toxic, assume first result is non-toxic, second has desired score.
    return [{"score": 0.1, "label": "non-toxic"}, {"score": 0.8, "label": "non-toxic"}]


# Fake pipeline for BiasMetric
def fake_pipeline_bias(text, top_k=None):
    # Returns a list of dicts with scores for 11 bias classes.
    return [
        {"score": 0.1},
        {"score": 0.2},
        {"score": 0.3},
        {"score": 0.4},
        {"score": 0.5},
        {"score": 0.6},
        {"score": 0.7},
        {"score": 0.8},
        {"score": 0.9},
        {"score": 1.0},
        {"score": 1.1},
    ]


def test_toxicity_metric_calculate():
    metric = ToxicityMetric()
    metric.pipeline = fake_pipeline_toxic  # Override downloaded model
    # Test when label is 'toxic'
    score = metric.calculate("toxic text")
    assert score == 0.95
    # Test when label is not 'toxic'
    score = metric.calculate("non toxic")
    assert score == 0.8


def test_toxicity_metric_run():
    metric = ToxicityMetric()
    metric.pipeline = fake_pipeline_toxic
    data = {"output": ["toxic text", "non toxic"]}
    metric.run(data)
    # Check that toxicity column is added with expected scores
    assert "toxicity" in data
    assert data["toxicity"] == [0.95, 0.8]


def test_bias_metric_calculate():
    metric = BiasMetric()
    metric.pipeline = fake_pipeline_bias
    output = metric.calculate("any text")
    # Expect the fake_pipeline_bias output as is
    assert output == fake_pipeline_bias("any text")


def test_bias_metric_run():
    metric = BiasMetric()
    metric.pipeline = fake_pipeline_bias
    data = {"output": ["text1", "text2"]}
    metric.run(data)
    # For each bias class, check that a column is added with proper values.
    for idx, class_name in enumerate(metric.classes):
        col = f"bias_{class_name}"
        assert col in data
        expected = [
            fake_pipeline_bias("text1")[idx]["score"],
            fake_pipeline_bias("text2")[idx]["score"],
        ]
        assert data[col] == expected
