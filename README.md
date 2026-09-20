# Fixed FastAPI Backend

This backend is prepared for deployment with the MobileNetV2 model.

Main endpoint:
`POST /predict`

Health check:
`GET /health`

The endpoint expects a multipart upload with field name `file`.

The current Keras model exposes one sigmoid output, so:
- `raw_score` = malignant probability
- `1 - raw_score` = benign probability

For frontend compatibility, the API also returns:
`raw_scores.benign`, `raw_scores.invalid`, and `raw_scores.malignant`.

Invalid or unreadable images return an `invalid` prediction with a suitable HTTP 400/413 response.

Do not put Back4App secrets in the source. Add them as Render environment variables.
