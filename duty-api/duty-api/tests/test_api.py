VALID = {
    "reflux_ratio": 2.6,
    "feed_total_kg_h": 185200.0,
    "P_cond_barg": 18.5,
    "T16_C": 144.0,
    "Q_cond_kW": 21000.0,
    "Q_reb_kW": 11000.0,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "T16_C" in body["features"]
    assert set(body["example"]) == set(body["features"])


def test_predict_ok(client):
    r = client.post("/predict", json={"features": VALID})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["duty_HT1_kW"], float)
    assert body["model_version"] == "test-0001"


def test_predict_missing_feature(client):
    bad = dict(VALID)
    bad.pop("Q_cond_kW")
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422
    assert "Q_cond_kW" in str(r.json())


def test_predict_out_of_range(client):
    r = client.post("/predict", json={"features": {**VALID, "T16_C": 400.0}})
    assert r.status_code == 422


def test_predict_unknown_feature_warns(client):
    r = client.post("/predict", json={"features": {**VALID, "foo": 1.0}})
    assert r.status_code == 200
    assert any("foo" in w for w in r.json()["warnings"])


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"items": [VALID, {**VALID, "T16_C": 146.0}]})
    assert r.status_code == 200
    preds = r.json()["duty_HT1_kW"]
    assert len(preds) == 2 and preds[1] > preds[0]
