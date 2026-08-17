# MVP API Scope

Keep the API small.

## Initial application API

```text
POST   /api/v1/auth/login

POST   /api/v1/analyses
GET    /api/v1/analyses
GET    /api/v1/analyses/{id}
GET    /api/v1/analyses/{id}/signals
GET    /api/v1/analyses/{id}/report
```

Additional endpoints should be added only when a concrete UI or public API requirement needs them.

## Avoid during MVP

```text
GraphQL
public gRPC
WebSocket
webhooks
generated SDKs
generic bulk API
```

Polling analysis status is acceptable initially.
