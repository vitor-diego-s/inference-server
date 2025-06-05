import os
import numpy as np
import cv2
import pytest
from unittest import mock

import processor

class TestRunInference:
    @pytest.fixture
    def dummy_image_bytes(self):

        image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if not os.path.exists(image_path):
            pytest.skip("Real image file not found for integration test.")
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return image_bytes

    @pytest.fixture
    def mock_model(self):
        mock_result = mock.Mock()
        mock_result.tojson.return_value = '{"predictions": []}'
        mock_model = mock.Mock(return_value=[mock_result])
        return mock_model, mock_result

    @staticmethod
    def plot_results_on_image(image_bytes, results, output_path=None):
        """
        Plots YOLO results on the image and returns the image as bytes (JPEG).
        Optionally writes the plotted image to disk if output_path is provided.
        """
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if hasattr(results, "plot"):
            plotted = results.plot()
        else:
            plotted = image
        _, buffer = cv2.imencode('.jpg', plotted)
        img_bytes = buffer.tobytes()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(img_bytes)
        return img_bytes

    def test_run_inference_calls_model_and_returns_json(self, monkeypatch, dummy_image_bytes, mock_model):
        mock_model_instance, mock_result = mock_model
        monkeypatch.setattr(processor, "model", mock_model_instance)

        _, result_json = processor.run_inference(dummy_image_bytes)
        assert result_json == '{"predictions": []}'
        mock_model_instance.assert_called_once()
        mock_result.tojson.assert_called_once()

    def test_run_inference_handles_invalid_image_bytes(self, monkeypatch):
        # Patch model to ensure it's not called
        monkeypatch.setattr(processor, "model", mock.Mock())
        # Pass invalid bytes
        with pytest.raises(Exception):
            processor.run_inference(b"not_an_image")

    def test_run_inference_model_returns_empty_list(self, monkeypatch, dummy_image_bytes):
        mock_model_instance = mock.Mock(return_value=[])
        monkeypatch.setattr(processor, "model", mock_model_instance)
        # Should raise IndexError when accessing results[0]
        with pytest.raises(IndexError):
            processor.run_inference(dummy_image_bytes)

    def test_run_inference_model_result_tojson_raises(self, monkeypatch, dummy_image_bytes):
        mock_result = mock.Mock()
        mock_result.tojson.side_effect = RuntimeError("tojson failed")
        mock_model_instance = mock.Mock(return_value=[mock_result])
        monkeypatch.setattr(processor, "model", mock_model_instance)
        with pytest.raises(RuntimeError, match="tojson failed"):
            processor.run_inference(dummy_image_bytes)

    @pytest.mark.integration
    def test_run_inference_with_real_image(self, dummy_image_bytes):

        results, result_json = processor.run_inference(dummy_image_bytes)
        assert isinstance(result_json, str)
        assert result_json

        print("Inference result JSON:", result_json)

        self.plot_results_on_image(dummy_image_bytes, results[0], "output_test_image.jpg")