# Model metrics

The `/metrics` endpoint computes the character error rate (CER) of the trained
ASR model on the validation dataset specified in `service_config.yaml`.

Example response:

```json
{"cer": 0.12}
```
