import os
import numpy as np
import cv2
import pytest
from unittest import mock

import src.processor as processor

image_ref = "https://images.unsplash.com/photo-1514510316197-ac2d9e62319a?fm=jpg"
process_ref = "test_process_123"

class TestRunInference:
    @pytest.fixture
    def dummy_image_bytes(self):

        image_bytes = processor.download_image_from_link(image_ref)
        if not image_bytes:
            raise ValueError("Failed to download image bytes for testing.")
        return image_bytes
    
    @pytest.fixture
    def dummy_process_payload(self):
        return {
            "link": image_ref,
            "process_id": process_ref
        }

    @pytest.fixture
    def mock_model(self):
        mock_result = mock.Mock()
        mock_result.to_json.return_value = '{"predictions": []}'
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

    def test_download_image_from_link(self, dummy_process_payload):
        image_link = dummy_process_payload["link"]
        image_bytes = processor.download_image_from_link(image_link)
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

    def test_run_inference_calls_model_and_returns_json(self, monkeypatch, dummy_image_bytes, mock_model):
        mock_model_instance, mock_result = mock_model
        monkeypatch.setattr(processor, "model", mock_model_instance)

        _, result_json = processor.execute(dummy_image_bytes)
        assert result_json == '{"predictions": []}'
        mock_model_instance.assert_called_once()
        mock_result.to_json.assert_called_once()

    def test_run_inference_handles_invalid_image_bytes(self, monkeypatch):
        # Patch model to ensure it's not called
        monkeypatch.setattr(processor, "model", mock.Mock())
        # Pass invalid bytes
        with pytest.raises(Exception):
            processor.execute(b"not_an_image")

    @pytest.mark.integration
    def test_run_inference(self, dummy_image_bytes):

        results, result_json = processor.execute(dummy_image_bytes)
        assert isinstance(result_json, str)
        assert result_json

        # print("Inference result JSON:", result_json)

        self.plot_results_on_image(dummy_image_bytes, results[0], "output_test_image.jpg")