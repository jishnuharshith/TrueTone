"""Check the JSON endpoints consumed by the integrated Android screens."""
from fastapi.testclient import TestClient
from backend.main import app, settings


def test_android_scripted_flow(monkeypatch):
    monkeypatch.setattr(settings, 'mock_mode', True)
    with TestClient(app) as client:
        assert client.get('/health').json()['status'] == 'ok'
        for text, risk in [
            ('Please tell me your OTP right now or your account will be blocked.', 'high'),
            ('Never share your OTP with anyone. Contact your bank directly.', 'low'),
        ]:
            response = client.post('/v1/demo/analyze-scripted-text', json={'text': text, 'language': 'auto'})
            assert response.status_code == 200
            data = response.json()
            assert data['risk_level'] == risk
            assert isinstance(data['evidence_labels'], list)
            assert 0 <= data['confidence'] <= 1
            assert data['model_version'] == 'text-prototype-v1'
            assert data['source'] == 'scripted_text_demo'
        assert client.get('/v1/demo/live-result').status_code == 200
        assert client.get('/demo').status_code == 200


def test_android_demo_disabled_is_not_a_risk_result(monkeypatch):
    monkeypatch.setattr(settings, 'mock_mode', False)
    with TestClient(app) as client:
        response = client.post('/v1/demo/analyze-scripted-text', json={'text': 'Test', 'language': 'auto'})
        assert response.status_code == 503
        assert 'risk_level' not in response.json()
