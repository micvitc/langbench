"""
Module for benchmarking language models.

This module provides an Evaluator class to run evaluations with various metrics.

"""

import time
import logging


from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


class Evaluator:
    """
    Evaluator class to execute language model evaluations using various metrics.

    Attributes:
        metrics (list): List of metric classes to evaluate outputs.
        online (bool): Flag indicating if evaluation should be run online using a pipeline.
        pipeline (callable): A callable that runs the model inference on input text.
    """

    def __init__(self, pipeline=None, online=False):
        """
        Initializes the Evaluator instance.

        Args:
            pipeline (callable, optional): A callable function for processing text.
                Required if 'online' is True.
            online (bool, optional): If True, executes pipeline processing during evaluation.

        Raises:
            ValueError: If 'online' is True but pipeline is not provided.
        """
        self.metrics = []
        self.online = online
        self.pipeline = pipeline
        if online and pipeline is None:
            raise ValueError("Pipeline must be provided for online evaluation")

    def add_metric(self, metric):
        """
        Adds a metric to the evaluator.

        Args:
            metric (class): A metric class to be added to the evaluator.
        """
        self.metrics.append(metric)

    def call_pipeline(self, text):
        """
        Calls the pipeline function on the provided text.

        Args:
            text (str): The input text to evaluate.

        Returns:
            str: The output content from the pipeline's processing.
        """
        return self.pipeline(text).content

    def execute(self, data):
        """
        Executes the pipeline on the 'input' column of the data frame and calculates latency.

        Args:
            data (DataFrame): A pandas DataFrame containing the 'input' column.
        """
        latencies = []

        def pipeline_with_latency(text):
            start_time = time.time()
            output = self.call_pipeline(text)
            end_time = time.time()
            latencies.append(end_time - start_time)
            return output

        data["output"] = data["input"].progress_map(pipeline_with_latency)
        data["latency"] = latencies

    def evaluate(self, input_data):
        """
        Evaluates the input data using all added metrics.

        Args:
            input_data (DataFrame): A pandas DataFrame with at least an 'input' column.

        Returns:
            DataFrame: A DataFrame containing original data along with evaluation metric outputs.
        """
        data = input_data.copy()
        if self.online:
            self.execute(data)
        for metric in self.metrics:
            m = metric()
            tqdm.write(f"Evaluating {m.name}")
            m.run(data)
            tqdm.write(f"Finished evaluating {m.name}")
        return data
