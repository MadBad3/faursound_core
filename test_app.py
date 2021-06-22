import pytest
from app import faursound_app

@pytest.fixture
def client():
    app = faursound_app()
    app.config['TESTING'] = True

    with app.app_context():
        with app.test_client() as client:
            yield client


def test_hello(client):
	response = client.get('/hello')
	assert response.status_code == 200
